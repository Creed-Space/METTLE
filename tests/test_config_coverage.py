"""Tests for config.py allowed-origin parsing and production validation."""

import warnings
from pathlib import Path
from typing import Any, cast

import pytest

from config import Settings

SettingsFactory = cast(Any, Settings)

PRODUCTION_CONFIG = {
    "environment": "production",
    "allowed_origins": "https://mettle.sh",
    "secret_key": "s" * 32,
    "admin_api_key": "a" * 32,
    "vcp_signing_key": "test-pem",
    "use_database": True,
    "database_url": "postgresql://db.example/mettle",
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
        "METTLE_ADMIN_API_KEY",
        "METTLE_VCP_SIGNING_KEY",
        "METTLE_VCP_SIGNING_KEY_ID",
        "METTLE_USE_DATABASE",
        "METTLE_DATABASE_URL",
    ):
        assert f"key: {key}" in blueprint
    assert "--workers 1" in blueprint
