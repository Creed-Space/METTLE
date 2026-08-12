"""Compare live Render service settings with the reviewed repository contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
RENDER_API = "https://api.render.com/v1"


@dataclass(frozen=True)
class Check:
    field: str
    expected: object
    observed: object

    @property
    def passed(self) -> bool:
        return self.expected == self.observed

    def as_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "status": "pass" if self.passed else "drift",
            "expected": self.expected,
            "observed": self.observed,
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(blueprint_path: Path, deployment_path: Path) -> dict[str, Any]:
    """Load the declarative service contract without resolving any secrets."""
    blueprint = yaml.safe_load(blueprint_path.read_text(encoding="utf-8"))
    deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
    authored_services = blueprint["services"]
    authored = {service["name"]: service for service in authored_services}
    if len(authored) != len(authored_services):
        raise ValueError("blueprint contains duplicate service names")
    bindings = deployment["services"]
    binding_names = [binding["blueprint_name"] for binding in bindings]
    if len(set(binding_names)) != len(binding_names):
        raise ValueError("deployment contains duplicate service bindings")
    services: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        name = binding["blueprint_name"]
        if name not in authored:
            raise ValueError(f"deployment binding has no blueprint service: {name}")
        service = authored[name]
        if service.get("repo") != deployment["repository"]:
            raise ValueError(
                f"repository identity differs for blueprint service: {name}"
            )
        nonsecret_env: dict[str, str] = {}
        secret_keys: set[str] = set(binding.get("provider_secret_keys", []))
        for declaration in service.get("envVars", []):
            key = declaration["key"]
            if "value" in declaration:
                nonsecret_env[key] = str(declaration["value"])
            elif (
                declaration.get("sync") is False
                or declaration.get("generateValue") is True
            ):
                secret_keys.add(key)
            else:
                raise ValueError(f"environment declaration is ambiguous: {name}.{key}")
        overlap = set(nonsecret_env) & secret_keys
        if overlap:
            raise ValueError(
                f"environment keys are both public and secret for {name}: {sorted(overlap)}"
            )
        services[name] = {
            "binding": binding,
            "blueprint": service,
            "nonsecret_env": nonsecret_env,
            "secret_keys": sorted(secret_keys),
        }
    if set(services) != set(authored):
        missing = sorted(set(authored) - set(services))
        raise ValueError(f"blueprint services lack production bindings: {missing}")
    return {
        "repository": deployment["repository"],
        "services": services,
        "blueprint_sha256": _sha256(blueprint_path),
        "deployment_sha256": _sha256(deployment_path),
    }


def _request(path: str, token: str) -> object:
    if not path.startswith("/") or "://" in path:
        raise ValueError("Render API path must stay under the fixed HTTPS origin")
    request = urllib.request.Request(
        f"{RENDER_API}{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "METTLE-Render-Drift-Check/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310
        return json.load(response)


def fetch_live(contract: dict[str, Any], token: str) -> dict[str, Any]:
    """Read the bound services and environment values from Render."""
    owners_payload = _request("/owners?limit=100", token)
    if not isinstance(owners_payload, list):
        raise ValueError("unexpected Render owners response")
    owners = {}
    for item in owners_payload:
        owner = item.get("owner", item)
        owners[owner["id"]] = owner["name"]

    services: dict[str, dict[str, Any]] = {}
    for name, expected in contract["services"].items():
        service_id = expected["binding"]["service_id"]
        service = _request(f"/services/{service_id}", token)
        env_payload = _request(f"/services/{service_id}/env-vars?limit=100", token)
        if not isinstance(service, dict) or not isinstance(env_payload, list):
            raise ValueError(f"unexpected Render response for {name}")
        environment = {}
        for item in env_payload:
            declaration = item.get("envVar", item)
            environment[declaration["key"]] = declaration.get("value", "")
        services[name] = {"service": service, "environment": environment}
    return {"owners": owners, "services": services}


def evaluate_drift(contract: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    """Return a receipt containing only nonsecret values and secret presence."""
    service_receipts = []
    all_checks: list[Check] = []
    for name, expected in sorted(contract["services"].items()):
        binding = expected["binding"]
        blueprint = expected["blueprint"]
        current = live["services"][name]
        service = current["service"]
        details = service.get("serviceDetails", {})
        runtime = details.get("envSpecificDetails", {})
        environment = current["environment"]
        checks = [
            Check("service_id", binding["service_id"], service.get("id")),
            Check("name", name, service.get("name")),
            Check("type", "web_service", service.get("type")),
            Check("workspace_id", binding["workspace_id"], service.get("ownerId")),
            Check(
                "workspace_name",
                binding["workspace_name"],
                live["owners"].get(service.get("ownerId")),
            ),
            Check("repository", contract["repository"], service.get("repo")),
            Check("branch", blueprint["branch"], service.get("branch")),
            Check(
                "auto_deploy",
                "yes" if blueprint["autoDeploy"] else "no",
                service.get("autoDeploy"),
            ),
            Check("runtime", blueprint["runtime"], details.get("runtime")),
            Check("plan", blueprint["plan"], details.get("plan")),
            Check("region", blueprint["region"], details.get("region")),
            Check(
                "num_instances", blueprint["numInstances"], details.get("numInstances")
            ),
            Check(
                "health_check_path",
                blueprint["healthCheckPath"],
                details.get("healthCheckPath"),
            ),
        ]
        if blueprint["runtime"] == "python":
            checks.extend(
                [
                    Check(
                        "build_command",
                        blueprint["buildCommand"],
                        runtime.get("buildCommand"),
                    ),
                    Check(
                        "start_command",
                        blueprint["startCommand"],
                        runtime.get("startCommand"),
                    ),
                ]
            )
        elif blueprint["runtime"] == "docker":
            checks.extend(
                [
                    Check(
                        "dockerfile_path",
                        blueprint["dockerfilePath"],
                        runtime.get("dockerfilePath"),
                    ),
                    Check(
                        "docker_context",
                        blueprint["dockerContext"],
                        runtime.get("dockerContext"),
                    ),
                    Check("docker_command", "", runtime.get("dockerCommand", "")),
                ]
            )
        else:
            raise ValueError(f"unsupported Render runtime: {blueprint['runtime']}")

        nonsecret = expected["nonsecret_env"]
        secret_keys = expected["secret_keys"]
        for key, value in sorted(nonsecret.items()):
            checks.append(Check(f"env.{key}", value, environment.get(key)))
        for key in secret_keys:
            checks.append(
                Check(
                    f"secret.{key}",
                    "present",
                    "present" if bool(environment.get(key)) else "missing",
                )
            )
        expected_keys = set(nonsecret) | set(secret_keys)
        checks.append(
            Check("environment_keys", sorted(expected_keys), sorted(environment))
        )
        all_checks.extend(checks)
        service_receipts.append(
            {
                "name": name,
                "service_id": binding["service_id"],
                "provider_updated_at": service.get("updatedAt"),
                "status": "pass" if all(check.passed for check in checks) else "drift",
                "checks": [check.as_dict() for check in checks],
            }
        )

    return {
        "schema_version": "1.0",
        "checked_at": datetime.now(UTC).isoformat(),
        "result": "match" if all(check.passed for check in all_checks) else "drift",
        "blueprint_sha256": contract["blueprint_sha256"],
        "deployment_sha256": contract["deployment_sha256"],
        "services": service_receipts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blueprint", type=Path, default=ROOT / "render.yaml")
    parser.add_argument(
        "--deployment", type=Path, default=ROOT / "deploy/render-production.json"
    )
    parser.add_argument("--token-stdin", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.token_stdin:
        parser.error("pass the Render API token through --token-stdin")
    token = sys.stdin.read().strip()
    if not token:
        parser.error("Render API token input is empty")
    exit_status = 0
    try:
        contract = load_contract(args.blueprint, args.deployment)
        receipt = evaluate_drift(contract, fetch_live(contract, token))
    except (OSError, ValueError, KeyError, urllib.error.URLError) as error:
        receipt = {
            "schema_version": "1.0",
            "checked_at": datetime.now(UTC).isoformat(),
            "result": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        exit_status = 2
    serialized = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if exit_status == 0 and receipt["result"] != "match":
        exit_status = 1
    raise SystemExit(exit_status)


if __name__ == "__main__":
    main()
