"""Tests for the secret-safe Render configuration drift check."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest

import scripts.check_render_drift as render_drift
from scripts.check_render_drift import evaluate_drift, load_contract


def _contract_files(tmp_path: Path) -> tuple[Path, Path]:
    blueprint = tmp_path / "render.yaml"
    blueprint.write_text(
        """services:
  - type: web
    name: service
    runtime: python
    repo: https://github.com/example/repo
    branch: main
    plan: starter
    region: oregon
    numInstances: 1
    buildCommand: pip install .
    startCommand: python main.py
    healthCheckPath: /health
    autoDeploy: true
    envVars:
      - key: PUBLIC_VALUE
        value: expected
      - key: REQUIRED_SECRET
        sync: false
""",
        encoding="utf-8",
    )
    deployment = tmp_path / "deployment.json"
    deployment.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "repository": "https://github.com/example/repo",
                "services": [
                    {
                        "blueprint_name": "service",
                        "service_id": "srv-1",
                        "workspace_id": "tea-1",
                        "workspace_name": "Production",
                        "provider_secret_keys": ["EXTRA_SECRET"],
                        "provider_secret_files": ["holder-policy.json"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return blueprint, deployment


def _live(*, build_command: str = "pip install .") -> dict:
    return {
        "owners": {"tea-1": "Production"},
        "services": {
            "service": {
                "service": {
                    "id": "srv-1",
                    "name": "service",
                    "type": "web_service",
                    "ownerId": "tea-1",
                    "repo": "https://github.com/example/repo",
                    "branch": "main",
                    "autoDeploy": "yes",
                    "updatedAt": "2026-08-12T20:00:00Z",
                    "serviceDetails": {
                        "runtime": "python",
                        "plan": "starter",
                        "region": "oregon",
                        "numInstances": 1,
                        "healthCheckPath": "/health",
                        "envSpecificDetails": {
                            "buildCommand": build_command,
                            "startCommand": "python main.py",
                        },
                    },
                },
                "environment": {
                    "PUBLIC_VALUE": "expected",
                    "REQUIRED_SECRET": "never-print-this-secret",  # pragma: allowlist secret
                    "EXTRA_SECRET": "never-print-this-either",  # pragma: allowlist secret
                },
                "secret_files": {
                    "holder-policy.json": "never-print-file-secret"  # pragma: allowlist secret
                },
            }
        },
    }


def _fingerprints(contract: dict, live: dict) -> dict[str, str]:
    fingerprints = {
        f"{name}.{key}": hashlib.sha256(
            live["services"][name]["environment"][key].encode()
        ).hexdigest()
        for name, service in contract["services"].items()
        for key in service["secret_keys"]
    }
    fingerprints.update(
        {
            f"{name}.secret_file:{filename}": hashlib.sha256(
                live["services"][name]["secret_files"][filename].encode()
            ).hexdigest()
            for name, service in contract["services"].items()
            for filename in service["secret_files"]
        }
    )
    return fingerprints


def test_matching_provider_contract_is_green_and_secret_safe(tmp_path: Path) -> None:
    blueprint, deployment = _contract_files(tmp_path)
    contract = load_contract(blueprint, deployment)
    live = _live()
    receipt = evaluate_drift(contract, live, _fingerprints(contract, live))
    serialized = json.dumps(receipt)
    assert receipt["result"] == "match"
    assert "never-print-this-secret" not in serialized
    assert "never-print-this-either" not in serialized
    assert "never-print-file-secret" not in serialized
    assert '"observed": "match"' in serialized


def test_substituted_nonempty_secret_is_detected_without_disclosing_it(
    tmp_path: Path,
) -> None:
    blueprint, deployment = _contract_files(tmp_path)
    contract = load_contract(blueprint, deployment)
    expected_live = _live()
    fingerprints = _fingerprints(contract, expected_live)
    substituted = _live()
    substituted["services"]["service"]["environment"]["REQUIRED_SECRET"] = (
        "different-still-nonempty-secret"  # pragma: allowlist secret
    )
    receipt = evaluate_drift(contract, substituted, fingerprints)
    serialized = json.dumps(receipt)
    assert receipt["result"] == "drift"
    secret_check = next(
        check
        for check in receipt["services"][0]["checks"]
        if check["field"] == "secret_identity.REQUIRED_SECRET"
    )
    assert secret_check["observed"] == "mismatch"
    assert "different-still-nonempty-secret" not in serialized


def test_substituted_secret_file_is_detected_without_disclosing_it(
    tmp_path: Path,
) -> None:
    blueprint, deployment = _contract_files(tmp_path)
    contract = load_contract(blueprint, deployment)
    expected_live = _live()
    fingerprints = _fingerprints(contract, expected_live)
    substituted = _live()
    substituted["services"]["service"]["secret_files"]["holder-policy.json"] = (
        "substituted-file-secret"  # pragma: allowlist secret
    )

    receipt = evaluate_drift(contract, substituted, fingerprints)
    serialized = json.dumps(receipt)
    check = next(
        item
        for item in receipt["services"][0]["checks"]
        if item["field"] == "secret_file_identity.holder-policy.json"
    )
    assert receipt["result"] == "drift"
    assert check["observed"] == "mismatch"
    assert "substituted-file-secret" not in serialized


def test_deliberate_build_command_mismatch_is_detected(tmp_path: Path) -> None:
    blueprint, deployment = _contract_files(tmp_path)
    contract = load_contract(blueprint, deployment)
    live = _live(build_command="pip install without hashes")
    receipt = evaluate_drift(contract, live, _fingerprints(contract, live))
    assert receipt["result"] == "drift"
    build_check = next(
        check
        for check in receipt["services"][0]["checks"]
        if check["field"] == "build_command"
    )
    assert build_check == {
        "field": "build_command",
        "status": "drift",
        "expected": "pip install .",
        "observed": "pip install without hashes",
    }


def test_contract_rejects_blueprint_repository_mismatch(tmp_path: Path) -> None:
    blueprint, deployment = _contract_files(tmp_path)
    blueprint.write_text(
        blueprint.read_text().replace(
            "https://github.com/example/repo", "https://github.com/attacker/repo"
        )
    )
    with pytest.raises(ValueError, match="repository identity differs"):
        load_contract(blueprint, deployment)


def test_cli_preserves_secret_safe_failure_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blueprint, deployment = _contract_files(tmp_path)
    output = tmp_path / "failure.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_render_drift.py",
            "--blueprint",
            str(blueprint),
            "--deployment",
            str(deployment),
            "--token-stdin",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO("secret-token"))
    contract = load_contract(blueprint, deployment)
    live = _live()
    monkeypatch.setenv(
        render_drift.SECRET_FINGERPRINT_ENV,
        json.dumps(
            {
                "schema_version": "1.0",
                "fingerprints": _fingerprints(contract, live),
            }
        ),
    )
    monkeypatch.setattr(
        render_drift,
        "fetch_live",
        lambda contract, token: (_ for _ in ()).throw(OSError("provider unavailable")),
    )

    with pytest.raises(SystemExit) as stopped:
        render_drift.main()
    assert stopped.value.code == 2
    receipt = json.loads(output.read_text())
    assert receipt["result"] == "error"
    assert receipt["error_type"] == "OSError"
    assert "secret-token" not in output.read_text()


def test_additional_holder_blueprint_must_have_a_deployment_binding(
    tmp_path: Path,
) -> None:
    blueprint, deployment = _contract_files(tmp_path)
    holder = tmp_path / "holder.yaml"
    holder.write_text(
        """services:
  - type: pserv
    name: holder
    runtime: python
    repo: https://github.com/example/repo
    branch: main
    plan: starter
    region: oregon
    numInstances: 1
    buildCommand: pip install --require-hashes -r requirements-production.txt
    startCommand: python holder.py
    maxShutdownDelaySeconds: 60
    autoDeploy: false
    disk:
      name: holder-fence
      mountPath: /var/lib/holder
      sizeGB: 1
    envVars:
      - key: HOLDER_SECRET
        sync: false
""",
        encoding="utf-8",
    )
    payload = json.loads(deployment.read_text())
    payload["required_blueprints"] = ["holder.yaml"]
    payload["services"].append(
        {
            "blueprint_name": "holder",
            "blueprint_path": "holder.yaml",
            "service_id": "srv-holder",
            "workspace_id": "tea-1",
            "workspace_name": "Production",
            "provider_secret_keys": [],
        }
    )
    deployment.write_text(json.dumps(payload))
    contract = load_contract(blueprint, deployment)
    assert set(contract["services"]) == {"service", "holder"}
    assert "holder.yaml" in contract["blueprint_sha256"]

    payload["services"].pop()
    deployment.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="lack deployment bindings"):
        load_contract(blueprint, deployment)


def test_render_checker_rejects_redirects_before_forwarding_bearer() -> None:
    with pytest.raises(urllib.error.HTTPError, match="redirect rejected"):
        render_drift._RejectRedirects().redirect_request(
            SimpleNamespace(full_url="https://api.render.com/v1/services"),
            None,
            307,
            "Temporary Redirect",
            {},
            "https://attacker.example/capture",
        )
