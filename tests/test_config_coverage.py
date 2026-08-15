"""Tests for config.py allowed-origin parsing and production validation."""

import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from config import Settings

SettingsFactory = cast(Any, Settings)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently accepting the last value."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)

PRODUCTION_CONFIG = {
    "environment": "production",
    "allowed_origins": "https://mettle.sh",
    "trusted_hosts": "mettle.sh",
    "secret_key": "s" * 32,
    "admin_api_key": "a" * 32,
    "vcp_signing_key": "test-pem",
    "use_database": True,
    "database_url": "postgresql://db.example/mettle?sslmode=verify-full",
    "redis_url": "rediss://redis.example/mettle",
}


class TestAllowedOriginsList:
    def test_wildcard_returns_single_star(self):
        """allowed_origins='*' returns ['*']."""
        s = SettingsFactory(allowed_origins="*", _env_file="nonexistent.env")
        assert s.allowed_origins_list == ["*"]

    def test_comma_separated_origins(self):
        """Comma-separated origins are split and stripped (line 78)."""
        s = SettingsFactory(
            allowed_origins="http://localhost:3000, https://example.com , https://other.io",
            _env_file="nonexistent.env",
        )
        assert s.allowed_origins_list == [
            "http://localhost:3000",
            "https://example.com",
            "https://other.io",
        ]

    def test_single_origin(self):
        """Single non-wildcard origin returns list of one."""
        s = SettingsFactory(
            allowed_origins="https://example.com", _env_file="nonexistent.env"
        )
        assert s.allowed_origins_list == ["https://example.com"]

    def test_trusted_hosts_are_split_and_stripped(self):
        s = SettingsFactory(
            trusted_hosts="mettle.sh, www.mettle.sh",
            _env_file="nonexistent.env",
        )
        assert s.trusted_hosts_list == ["mettle.sh", "www.mettle.sh"]

    def test_trusted_hosts_wildcard_is_preserved(self):
        s = SettingsFactory(trusted_hosts="*", _env_file="nonexistent.env")
        assert s.trusted_hosts_list == ["*"]


class TestProductionValidation:
    def test_production_wildcard_cors_rejected(self):
        """Production + wildcard CORS is rejected."""
        with pytest.raises(ValueError, match="trusted origins"):
            SettingsFactory(
                **{**PRODUCTION_CONFIG, "allowed_origins": "*"},
                _env_file="nonexistent.env",
            )

    def test_production_specific_origins_no_warning(self):
        """Production with specific origins does not warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SettingsFactory(
                **PRODUCTION_CONFIG,
                _env_file="nonexistent.env",
            )
            security_warnings = [x for x in w if "SECURITY WARNING" in str(x.message)]
            assert len(security_warnings) == 0

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            ({"secret_key": "short"}, "SECRET_KEY"),
            ({"admin_api_key": "short"}, "ADMIN_API_KEY"),
            ({"vcp_signing_key": ""}, "VCP_SIGNING_KEY"),
            ({"use_database": False}, "USE_DATABASE"),
            ({"database_url": "sqlite:///mettle.db"}, "PostgreSQL"),
            ({"database_url": "postgresql://db.example/mettle"}, "sslmode"),
            ({"allowed_origins": "http://mettle.sh"}, "HTTPS"),
            ({"trusted_hosts": "*"}, "TRUSTED_HOSTS"),
            ({"redis_url": ""}, "REDIS_URL"),
            ({"redis_url": "redis://redis.example/mettle"}, "rediss"),
        ],
    )
    def test_insecure_production_settings_rejected(self, override, message):
        config = {**PRODUCTION_CONFIG, **override}
        with pytest.raises(ValueError, match=message):
            SettingsFactory(**config, _env_file="nonexistent.env")

    @pytest.mark.parametrize(
        "redis_url",
        [
            "rediss://redis.example/mettle?ssl_cert_reqs=none",
            "rediss://redis.example/mettle?ssl_check_hostname=false",
            ("rediss://redis.example/mettle?ssl_cert_reqs=required&ssl_cert_reqs=none"),
            (
                "rediss://redis.example/mettle?ssl_check_hostname=true"
                "&ssl_check_hostname=false"
            ),
        ],
    )
    def test_redis_tls_query_cannot_downgrade_verification(self, redis_url):
        with pytest.raises(ValueError, match="TLS (certificate|hostname)"):
            SettingsFactory(
                **{**PRODUCTION_CONFIG, "redis_url": redis_url},
                _env_file="nonexistent.env",
            )

    @pytest.mark.parametrize(
        "redis_url",
        [
            "rediss://redis.example/mettle",
            (
                "rediss://redis.example/mettle?ssl_cert_reqs=required"
                "&ssl_check_hostname=true"
            ),
            "rediss://redis.example/mettle?ssl_cert_reqs=cert_required",
        ],
    )
    def test_redis_tls_query_accepts_required_verification(self, redis_url):
        settings = SettingsFactory(
            **{**PRODUCTION_CONFIG, "redis_url": redis_url},
            _env_file="nonexistent.env",
        )
        assert settings.redis_url == redis_url

    def test_development_wildcard_no_warning(self):
        """Development mode with wildcard does not warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SettingsFactory(
                environment="development",
                allowed_origins="*",
                _env_file="nonexistent.env",
            )
            security_warnings = [x for x in w if "SECURITY WARNING" in str(x.message)]
            assert len(security_warnings) == 0


def test_render_blueprint_declares_fail_closed_production_dependencies():
    """Render must supply every setting required by production validation."""
    blueprint = (Path(__file__).parent.parent / "render.yaml").read_text()
    for key in (
        "METTLE_ALLOWED_ORIGINS",
        "METTLE_TRUSTED_HOSTS",
        "METTLE_CREDENTIAL_ISSUANCE_ENABLED",
        "METTLE_ADMIN_API_KEY",
        "METTLE_VCP_SIGNING_KEY",
        "METTLE_VCP_SIGNING_KEY_ID",
        "METTLE_USE_DATABASE",
        "METTLE_DATABASE_URL",
        "METTLE_REDIS_URL",
        "METTLE_FORWARDED_ALLOW_IPS",
    ):
        assert f"key: {key}" in blueprint
    assert "--workers 2" in blueprint
    parsed = yaml.safe_load(blueprint)
    env = {item["key"]: item for item in parsed["services"][0]["envVars"]}
    assert env["METTLE_FORWARDED_ALLOW_IPS"] == {
        "key": "METTLE_FORWARDED_ALLOW_IPS",
        "value": "10.0.0.0/8",
    }


def test_render_proxy_trust_ignores_caller_prepended_forwarded_identity() -> None:
    """Only the Render ingress peer may contribute a forwarded client chain."""
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    blueprint = yaml.safe_load(
        (Path(__file__).parent.parent / "render.yaml").read_text()
    )
    env = {
        item["key"]: item["value"]
        for item in blueprint["services"][0]["envVars"]
        if "value" in item
    }
    app = FastAPI()

    @app.get("/")
    async def remote_address(request: Request) -> dict[str, str]:
        assert request.client is not None
        return {"client": request.client.host}

    protected = ProxyHeadersMiddleware(
        cast(Any, app),
        trusted_hosts=env["METTLE_FORWARDED_ALLOW_IPS"],
    )
    forwarded = "198.51.100.44, 203.0.113.9"
    with TestClient(cast(Any, protected), client=("10.233.22.235", 50000)) as ingress:
        assert ingress.get("/", headers={"X-Forwarded-For": forwarded}).json() == {
            "client": "203.0.113.9"
        }
    with TestClient(cast(Any, protected), client=("192.0.2.5", 50000)) as direct:
        assert direct.get("/", headers={"X-Forwarded-For": forwarded}).json() == {
            "client": "192.0.2.5"
        }


def test_configured_database_import_failure_stops_application_startup() -> None:
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'database':
        raise ImportError('database unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import main
"""
    env = os.environ.copy()
    env.update(
        {
            "METTLE_ENVIRONMENT": "development",
            "METTLE_USE_DATABASE": "true",
            "METTLE_DATABASE_URL": "postgresql://example.invalid/mettle",
        }
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode != 0
    assert "Configured database module is unavailable" in completed.stderr


def test_holder_blueprint_forces_stop_first_singleton_deploys() -> None:
    """Render must not overlap two processes for one holder identity."""
    path = Path(__file__).parent.parent / "deploy" / "holder" / "render.yaml"
    blueprint = yaml.safe_load(path.read_text())
    service = blueprint["services"][0]
    assert service["type"] == "pserv"
    assert service["maxShutdownDelaySeconds"] == 60
    assert service["disk"] == {
        "name": "mettle-holder-singleton-fence",
        "mountPath": "/var/lib/mettle-holder",
        "sizeGB": 1,
    }


def test_all_authored_yaml_rejects_duplicate_keys() -> None:
    """Workflow and deployment YAML must not hide values behind duplicate keys."""
    root = Path(__file__).parent.parent
    paths = [
        *sorted((root / ".github").rglob("*.yml")),
        *sorted((root / ".github").rglob("*.yaml")),
        root / "render.yaml",
        root / "deploy/holder/render.yaml",
    ]
    for path in paths:
        yaml.load(path.read_text(), Loader=_UniqueKeyLoader)


def test_tag_release_reuses_full_ci_on_the_exact_candidate() -> None:
    """A tag must pass the same candidate workflow before release publication."""
    root = Path(__file__).parent.parent
    ci = (root / ".github/workflows/ci.yml").read_text()
    release = (root / ".github/workflows/release.yml").read_text()
    assert "workflow_call:" in ci
    assert "uses: ./.github/workflows/ci.yml" in release
    assert "needs: validate-candidate" in release
    for workflow in (ci, release):
        assert "pip install --require-hashes" in workflow
        assert (
            "cyclonedx-py environment /tmp/mettle-production-lock/bin/python"
            in workflow
        )
        assert "scripts/finalize_server_sbom.py" in workflow
    assert "-r requirements-release-lock.txt" in release
    assert "pip install -r requirements-dev.txt" not in release


def test_security_gate_covers_release_code_and_every_workflow_lock() -> None:
    """New release authority and its dependency sets stay inside CI security scope."""
    ci = (Path(__file__).parent.parent / ".github/workflows/ci.yml").read_text()

    assert ci.count("scripts/deploy_render_release.py") == 2
    for lock in (
        "requirements-mcp-lock.txt",
        "requirements-build-lock.txt",
        "requirements-drift-lock.txt",
        "requirements-release-lock.txt",
    ):
        assert f"pip-audit -r {lock}" in ci


def test_release_requires_reproducibility_before_publication() -> None:
    """No public package may precede three-builder byte-identity proof."""
    root = Path(__file__).parent.parent
    ci = (root / ".github/workflows/ci.yml").read_text()
    release = (root / ".github/workflows/release.yml").read_text()

    assert "builder: [linux-1, linux-2]" in ci
    assert 'python-version: "3.13.14"' in ci
    assert "-m scripts.build_distributions" in ci
    assert "name: Reproducibility gate" in release
    assert (
        "validate-candidate:\n"
        "    permissions:\n"
        "      contents: read\n"
        "      checks: write\n"
        "    uses: ./.github/workflows/ci.yml"
    ) in release
    assert 'python-version: "3.13.14"' in release
    assert "--min-linux-builders 2 --require-macos" in release
    assert "name: Prepare exact release bundle" in release
    assert "needs: [validate-candidate, reproducibility, render-drift]" in release
    assert "Validate Official MCP Registry manifest before publication" in release
    assert "/tmp/mcp-publisher validate" in release
    assert release.index("/tmp/mcp-publisher validate") < release.index("publish-pypi:")
    assert "environment:\n      name: pypi" in release
    sha_prefix = "dc37677b2e1c63e2034f"  # pragma: allowlist secret
    sha_suffix = "94d8a5b11f265b73ba33"  # pragma: allowlist secret
    pypi_action_sha = sha_prefix + sha_suffix
    assert f"pypa/gh-action-pypi-publish@{pypi_action_sha}" in release
    assert "scripts/verify_pypi_release.py" in release
    assert "mcp-publisher publish" in release
    assert "scripts/build_distribution_receipt.py" in release
    assert "name: Publish complete GitHub Release" in release
    assert "needs: [prepare-release, verify-and-publish-registry]" in release
    assert "name: Render configuration drift gate" in release
    assert "scripts/check_render_drift.py" in release


def test_github_release_recovery_is_exact_and_cannot_republish_packages() -> None:
    """A failed final job can recover preserved artifacts without a second publish."""
    root = Path(__file__).parent.parent
    recovery = (root / ".github/workflows/recover-github-release.yml").read_text()
    manifest_builder = (root / "scripts/build_release_manifest.py").read_text()

    assert "workflow_dispatch:" in recovery
    assert "failed_release_run_id:" in recovery
    assert 'test "$(git rev-list -n 1 "$recovery_ref")" = "$SOURCE_SHA"' in recovery
    assert "release-prepared-${{ inputs.source_sha }}" in recovery
    assert "distribution-release-${{ inputs.source_sha }}" in recovery
    assert "run-id: ${{ inputs.failed_release_run_id }}" in recovery
    assert "Attest every recovered release artifact" in recovery
    assert "actions/attest-build-provenance@" in recovery
    assert "gh release create" in recovery
    assert "gh-action-pypi-publish" not in recovery
    assert "mcp-publisher publish" not in recovery
    assert "load_protocol_versions" in manifest_builder
    assert "from mettle.protocol import" not in manifest_builder


def test_render_drift_gate_is_read_only_and_scheduled() -> None:
    """Provider configuration has an exact, recurring, secret-safe check."""
    root = Path(__file__).parent.parent
    workflow = (root / ".github/workflows/render-drift.yml").read_text()
    contract = (root / "deploy/render-production.json").read_text()
    checker = (root / "scripts/check_render_drift.py").read_text()

    assert 'cron: "17 5 * * *"' in workflow
    assert "--token-stdin" in workflow
    assert "RENDER_API_TOKEN" in workflow
    assert '"srv-d5ujjr7pm1nc73bu5k3g"' in contract
    assert '"srv-d9h2p5beo5us73b4fh90"' in contract
    assert '"srv-d9b36jjeo5us73drljbg"' in contract
    assert '"holder/render.yaml"' in contract
    assert "RENDER_SECRET_FINGERPRINTS" in workflow
    assert "urllib.request.Request" in checker
    for mutating_method in ('method="POST"', 'method="PUT"', 'method="PATCH"'):
        assert mutating_method not in checker
