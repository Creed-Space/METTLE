"""Extended tests for config.py - covers production validation and missing lines."""

import warnings
from typing import Any, cast
from unittest.mock import patch

import pytest

from config import Settings, get_settings

SettingsFactory = cast(Any, Settings)

PRODUCTION_ENV = {
    "METTLE_ENVIRONMENT": "production",
    "METTLE_ALLOWED_ORIGINS": "https://mettle.sh",
    "METTLE_SECRET_KEY": "s" * 32,
    "METTLE_ADMIN_API_KEY": "a" * 32,
    "METTLE_VCP_SIGNING_KEY": "test-pem",
    "METTLE_USE_DATABASE": "true",
    "METTLE_DATABASE_URL": "postgresql://db.example/mettle",
}


class TestAllowedOriginsList:
    """Test allowed_origins_list property (covers line 78)."""

    def test_wildcard(self):
        with patch.dict("os.environ", {}, clear=True):
            s = SettingsFactory(_env_file=None)
        assert s.allowed_origins_list == ["*"]

    def test_comma_separated(self):
        with patch.dict("os.environ", {"METTLE_ALLOWED_ORIGINS": "http://a.com, http://b.com"}, clear=True):
            s = SettingsFactory(_env_file=None)
        assert s.allowed_origins_list == ["http://a.com", "http://b.com"]

    def test_single_origin(self):
        with patch.dict("os.environ", {"METTLE_ALLOWED_ORIGINS": "http://only.com"}, clear=True):
            s = SettingsFactory(_env_file=None)
        assert s.allowed_origins_list == ["http://only.com"]


class TestProductionValidation:
    """Test production config validator (covers lines 89-91)."""

    def test_production_wildcard_origins_rejected(self):
        env = {**PRODUCTION_ENV, "METTLE_ALLOWED_ORIGINS": "*"}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(ValueError, match="trusted origins"):
                SettingsFactory(_env_file=None)

    def test_production_specific_origins_no_warning(self):
        env = {**PRODUCTION_ENV, "METTLE_ALLOWED_ORIGINS": "https://app.creed.space"}
        with patch.dict("os.environ", env, clear=True):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                SettingsFactory(_env_file=None)
                security_warnings = [x for x in w if "SECURITY WARNING" in str(x.message)]
                assert len(security_warnings) == 0

    def test_development_wildcard_no_warning(self):
        env = {
            "METTLE_ENVIRONMENT": "development",
            "METTLE_ALLOWED_ORIGINS": "*",
        }
        with patch.dict("os.environ", env, clear=True):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                s = SettingsFactory(_env_file=None)
                assert s.is_production is False
                security_warnings = [x for x in w if "SECURITY WARNING" in str(x.message)]
                assert len(security_warnings) == 0


class TestIsProduction:
    """Test is_production property."""

    def test_production(self):
        with patch.dict(
            "os.environ",
            PRODUCTION_ENV,
            clear=True,
        ):
            s = SettingsFactory(_env_file=None)
        assert s.is_production is True

    def test_production_case_insensitive(self):
        with patch.dict(
            "os.environ",
            {**PRODUCTION_ENV, "METTLE_ENVIRONMENT": "Production"},
            clear=True,
        ):
            s = SettingsFactory(_env_file=None)
        assert s.is_production is True

    def test_development_not_production(self):
        with patch.dict("os.environ", {"METTLE_ENVIRONMENT": "development"}, clear=True):
            s = SettingsFactory(_env_file=None)
        assert s.is_production is False


class TestGetSettings:
    """Test get_settings cached factory."""

    def test_returns_settings_instance(self):
        get_settings.cache_clear()
        s = get_settings()
        assert isinstance(s, Settings)

    def test_cached(self):
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
