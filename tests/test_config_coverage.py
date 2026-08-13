"""Tests for config.py allowed-origin parsing and production validation."""

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
    "database_url": "postgresql://db.example/mettle",
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
            ({"allowed_origins": "http://mettle.sh"}, "HTTPS"),
            ({"trusted_hosts": "*"}, "TRUSTED_HOSTS"),
            ({"redis_url": ""}, "REDIS_URL"),
        ],
    )
    def test_insecure_production_settings_rejected(self, override, message):
        config = {**PRODUCTION_CONFIG, **override}
        with pytest.raises(ValueError, match=message):
            SettingsFactory(**config, _env_file="nonexistent.env")

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
    ):
        assert f"key: {key}" in blueprint
    assert "--workers 2" in blueprint


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


def test_release_requires_reproducibility_before_publication() -> None:
    """No public package may precede three-builder byte-identity proof."""
    root = Path(__file__).parent.parent
    ci = (root / ".github/workflows/ci.yml").read_text()
    release = (root / ".github/workflows/release.yml").read_text()

    assert "builder: [linux-1, linux-2]" in ci
    assert 'python-version: "3.13.14"' in ci
    assert "-m scripts.build_distributions" in ci
    assert "name: Reproducibility gate" in release
    assert 'python-version: "3.13.14"' in release
    assert "--min-linux-builders 2 --require-macos" in release
    assert "name: Prepare exact release bundle" in release
    assert "needs: [validate-candidate, reproducibility, render-drift]" in release
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
    assert "urllib.request.Request" in checker
    for mutating_method in ('method="POST"', 'method="PUT"', 'method="PATCH"'):
        assert mutating_method not in checker
