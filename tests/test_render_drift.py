"""Tests for the secret-safe Render configuration drift check."""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
            }
        },
    }


def test_matching_provider_contract_is_green_and_secret_safe(tmp_path: Path) -> None:
    blueprint, deployment = _contract_files(tmp_path)
    receipt = evaluate_drift(load_contract(blueprint, deployment), _live())
    serialized = json.dumps(receipt)
    assert receipt["result"] == "match"
    assert "never-print-this-secret" not in serialized
    assert "never-print-this-either" not in serialized
    assert '"observed": "present"' in serialized


def test_deliberate_build_command_mismatch_is_detected(tmp_path: Path) -> None:
    blueprint, deployment = _contract_files(tmp_path)
    receipt = evaluate_drift(
        load_contract(blueprint, deployment),
        _live(build_command="pip install without hashes"),
    )
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
