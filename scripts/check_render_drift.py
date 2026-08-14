"""Compare live Render service settings with the reviewed repository contract."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml


ROOT = Path(__file__).resolve().parents[1]
RENDER_API = "https://api.render.com/v1"
SECRET_FINGERPRINT_ENV = (
    "RENDER_SECRET_FINGERPRINTS"  # pragma: allowlist secret  # nosec B105
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _url_secret_semantics(key: str, value: object) -> str | None:
    """Return a secret-safe verdict for structured transport credentials."""
    if key not in {
        "METTLE_DATABASE_URL",
        "METTLE_HOLDER_DATABASE_URL",
        "METTLE_HOLDER_VAULT_URL",
        "METTLE_REDIS_URL",
    }:
        return None
    if not isinstance(value, str) or not value:
        return "mismatch"
    try:
        parsed = urllib.parse.urlsplit(value)
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        hostname = parsed.hostname
    except ValueError:
        return "mismatch"
    if not hostname:
        return "mismatch"
    if key in {"METTLE_DATABASE_URL", "METTLE_HOLDER_DATABASE_URL"}:
        ssl_modes = query.get("sslmode", [])
        valid = (
            parsed.scheme in {"postgres", "postgresql"}
            and len(ssl_modes) == 1
            and ssl_modes[0].strip().lower() == "verify-full"
        )
    elif key == "METTLE_REDIS_URL":
        certificate_requirements = query.get("ssl_cert_reqs", [])
        hostname_checks = query.get("ssl_check_hostname", [])
        valid = (
            parsed.scheme == "rediss"
            and (
                not certificate_requirements
                or (
                    len(certificate_requirements) == 1
                    and certificate_requirements[0].strip().lower()
                    in {"required", "cert_required"}
                )
            )
            and (
                not hostname_checks
                or (
                    len(hostname_checks) == 1
                    and hostname_checks[0].strip().lower() == "true"
                )
            )
        )
    else:
        valid = parsed.scheme == "https"
    return "match" if valid else "mismatch"


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Never forward the Render bearer token across an HTTP redirect."""

    def redirect_request(
        self,
        req: object,
        fp: object,
        code: int,
        _msg: str,
        headers: object,
        _newurl: str,
    ) -> None:
        raise urllib.error.HTTPError(
            getattr(req, "full_url", RENDER_API),
            code,
            "redirect rejected",
            cast(Any, headers),
            cast(Any, fp),
        )


_HTTPS_OPENER = urllib.request.build_opener(_RejectRedirects())


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


def _blueprint_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name


def _load_blueprint(path: Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    services = payload.get("services") if isinstance(payload, dict) else None
    if not isinstance(services, list) or not services:
        raise ValueError(f"blueprint has no services: {_blueprint_label(path)}")
    authored = {
        service["name"]: service for service in services if isinstance(service, dict)
    }
    if len(authored) != len(services):
        raise ValueError(
            f"blueprint has invalid or duplicate services: {_blueprint_label(path)}"
        )
    return authored


def load_contract(blueprint_path: Path, deployment_path: Path) -> dict[str, Any]:
    """Load the declarative service contract without resolving any secrets."""
    deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
    bindings = deployment["services"]
    binding_names = [binding["blueprint_name"] for binding in bindings]
    if len(set(binding_names)) != len(binding_names):
        raise ValueError("deployment contains duplicate service bindings")
    default_blueprint = blueprint_path.resolve()
    blueprints: dict[Path, dict[str, dict[str, Any]]] = {
        default_blueprint: _load_blueprint(default_blueprint)
    }
    for declared_path in deployment.get("required_blueprints", []):
        relative_path = Path(str(declared_path))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("additional blueprint path must be repository-relative")
        source_path = (deployment_path.parent / relative_path).resolve()
        blueprints[source_path] = _load_blueprint(source_path)
    services: dict[str, dict[str, Any]] = {}
    bound_by_blueprint: dict[Path, set[str]] = {path: set() for path in blueprints}
    for binding in bindings:
        relative_blueprint = binding.get("blueprint_path")
        if relative_blueprint is None:
            source_path = default_blueprint
        else:
            relative_path = Path(str(relative_blueprint))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    "additional blueprint path must be repository-relative"
                )
            source_path = (deployment_path.parent / relative_path).resolve()
            if source_path not in blueprints:
                blueprints[source_path] = _load_blueprint(source_path)
            bound_by_blueprint.setdefault(source_path, set())
        authored = blueprints[source_path]
        name = binding["blueprint_name"]
        if name not in authored:
            raise ValueError(f"deployment binding has no blueprint service: {name}")
        service = authored[name]
        if service.get("repo") != deployment["repository"]:
            raise ValueError(
                f"repository identity differs for blueprint service: {name}"
            )
        if name in services:
            raise ValueError(f"service name is repeated across blueprints: {name}")
        bound_by_blueprint[source_path].add(name)
        nonsecret_env: dict[str, str] = {}
        secret_keys: set[str] = set(binding.get("provider_secret_keys", []))
        raw_secret_files = binding.get("provider_secret_files", [])
        if not isinstance(raw_secret_files, list) or any(
            not isinstance(filename, str)
            or not filename.strip()
            or len(filename) > 255
            or Path(filename).name != filename
            or filename in {".", ".."}
            for filename in raw_secret_files
        ):
            raise ValueError(f"provider secret file names are invalid for {name}")
        if len(set(raw_secret_files)) != len(raw_secret_files):
            raise ValueError(f"provider secret file names are duplicated for {name}")
        secret_files = sorted(raw_secret_files)
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
            "blueprint_label": _blueprint_label(source_path),
            "nonsecret_env": nonsecret_env,
            "secret_keys": sorted(secret_keys),
            "secret_files": secret_files,
        }
    for source_path, authored in blueprints.items():
        missing = sorted(set(authored) - bound_by_blueprint[source_path])
        if missing:
            raise ValueError(
                f"blueprint services lack deployment bindings in "
                f"{_blueprint_label(source_path)}: {missing}"
            )
    return {
        "repository": deployment["repository"],
        "services": services,
        "blueprint_sha256": {
            _blueprint_label(path): _sha256(path) for path in sorted(blueprints)
        },
        "deployment_sha256": _sha256(deployment_path),
    }


def load_secret_fingerprints(
    serialized: str, contract: dict[str, Any]
) -> dict[str, str]:
    """Validate the encrypted, release-controlled secret identity inventory."""
    try:
        payload = json.loads(serialized)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("secret fingerprint inventory is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("secret fingerprint inventory schema is unsupported")
    fingerprints = payload.get("fingerprints")
    if not isinstance(fingerprints, dict):
        raise ValueError("secret fingerprint inventory is missing fingerprints")
    expected = {
        f"{name}.{key}"
        for name, service in contract["services"].items()
        for key in service["secret_keys"]
    }
    expected.update(
        f"{name}.secret_file:{filename}"
        for name, service in contract["services"].items()
        for filename in service["secret_files"]
    )
    if set(fingerprints) != expected:
        raise ValueError(
            "secret fingerprint inventory does not match reviewed secret keys"
        )
    for fingerprint in fingerprints.values():
        if not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(fingerprint):
            raise ValueError("secret fingerprint inventory contains an invalid digest")
    return dict(fingerprints)


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
    with _HTTPS_OPENER.open(request, timeout=30) as response:  # nosec B310
        final = urllib.parse.urlsplit(response.geturl())
        expected = urllib.parse.urlsplit(RENDER_API)
        if (final.scheme, final.netloc) != (expected.scheme, expected.netloc):
            raise ValueError("Render API response changed origin")
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
        secret_file_payload = _request(
            f"/services/{service_id}/secret-files?limit=100", token
        )
        if (
            not isinstance(service, dict)
            or not isinstance(env_payload, list)
            or not isinstance(secret_file_payload, list)
        ):
            raise ValueError(f"unexpected Render response for {name}")
        environment = {}
        for item in env_payload:
            declaration = item.get("envVar", item)
            if declaration["key"] in environment:
                raise ValueError(f"duplicate Render environment key for {name}")
            environment[declaration["key"]] = declaration.get("value", "")
        secret_files = {}
        for item in secret_file_payload:
            declaration = item.get("secretFile", item)
            if not isinstance(declaration, dict):
                raise ValueError(f"unexpected Render secret file for {name}")
            filename = declaration.get("name")
            content = declaration.get("content")
            if (
                not isinstance(filename, str)
                or not isinstance(content, str)
                or filename in secret_files
            ):
                raise ValueError(f"invalid or duplicate Render secret file for {name}")
            secret_files[filename] = content
        services[name] = {
            "service": service,
            "environment": environment,
            "secret_files": secret_files,
        }
    return {"owners": owners, "services": services}


def evaluate_drift(
    contract: dict[str, Any],
    live: dict[str, Any],
    secret_fingerprints: dict[str, str],
) -> dict[str, Any]:
    """Return a receipt containing nonsecret values and secret identity verdicts."""
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
        secret_files = current["secret_files"]
        service_type = {"web": "web_service", "pserv": "private_service"}.get(
            blueprint.get("type")
        )
        if service_type is None:
            raise ValueError(
                f"unsupported Render service type: {blueprint.get('type')}"
            )
        checks = [
            Check("service_id", binding["service_id"], service.get("id")),
            Check("name", name, service.get("name")),
            Check("type", service_type, service.get("type")),
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
        ]
        if "numInstances" in blueprint:
            checks.append(
                Check(
                    "num_instances",
                    blueprint["numInstances"],
                    details.get("numInstances"),
                )
            )
        if "healthCheckPath" in blueprint:
            checks.append(
                Check(
                    "health_check_path",
                    blueprint["healthCheckPath"],
                    details.get("healthCheckPath"),
                )
            )
        if "maxShutdownDelaySeconds" in blueprint:
            checks.append(
                Check(
                    "max_shutdown_delay_seconds",
                    blueprint["maxShutdownDelaySeconds"],
                    details.get("maxShutdownDelaySeconds"),
                )
            )
        if "disk" in blueprint:
            expected_disk = blueprint["disk"]
            observed_disk = details.get("disk") or {}
            for field in ("name", "mountPath", "sizeGB"):
                checks.append(
                    Check(
                        f"disk.{field}",
                        expected_disk.get(field),
                        observed_disk.get(field),
                    )
                )
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
            secret = environment.get(key)
            if not secret:
                identity = "missing"
            else:
                observed_fingerprint = hashlib.sha256(str(secret).encode()).hexdigest()
                identity = (
                    "match"
                    if hmac.compare_digest(
                        observed_fingerprint,
                        secret_fingerprints[f"{name}.{key}"],
                    )
                    else "mismatch"
                )
            checks.append(
                Check(
                    f"secret_identity.{key}",
                    "match",
                    identity,
                )
            )
            semantics = _url_secret_semantics(key, secret)
            if semantics is not None:
                checks.append(
                    Check(
                        f"secret_semantics.{key}",
                        "match",
                        semantics,
                    )
                )
        expected_keys = set(nonsecret) | set(secret_keys)
        checks.append(
            Check("environment_keys", sorted(expected_keys), sorted(environment))
        )
        for filename in expected["secret_files"]:
            content = secret_files.get(filename)
            if content is None:
                identity = "missing"
            else:
                observed_fingerprint = hashlib.sha256(content.encode()).hexdigest()
                identity = (
                    "match"
                    if hmac.compare_digest(
                        observed_fingerprint,
                        secret_fingerprints[f"{name}.secret_file:{filename}"],
                    )
                    else "mismatch"
                )
            checks.append(Check(f"secret_file_identity.{filename}", "match", identity))
        checks.append(
            Check(
                "secret_file_names",
                expected["secret_files"],
                sorted(secret_files),
            )
        )
        all_checks.extend(checks)
        service_receipts.append(
            {
                "name": name,
                "blueprint": expected["blueprint_label"],
                "service_id": binding["service_id"],
                "provider_updated_at": service.get("updatedAt"),
                "status": "pass" if all(check.passed for check in checks) else "drift",
                "checks": [check.as_dict() for check in checks],
            }
        )

    return {
        "schema_version": "1.1",
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
        fingerprints = load_secret_fingerprints(
            os.environ.get(SECRET_FINGERPRINT_ENV, ""), contract
        )
        receipt = evaluate_drift(
            contract,
            fetch_live(contract, token),
            fingerprints,
        )
    except (OSError, ValueError, KeyError, urllib.error.URLError) as error:
        receipt = {
            "schema_version": "1.1",
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
