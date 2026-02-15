"""Tests for config.py — coverage gaps on lines 78, 89-91."""

import warnings

import pytest

from config import Settings


class TestAllowedOriginsList:
    def test_wildcard_returns_single_star(self):
        """allowed_origins='*' returns ['*']."""
        s = Settings(allowed_origins="*", _env_file="nonexistent.env")
        assert s.allowed_origins_list == ["*"]

    def test_comma_separated_origins(self):
        """Comma-separated origins are split and stripped (line 78)."""
        s = Settings(
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
        s = Settings(allowed_origins="https://example.com", _env_file="nonexistent.env")
        assert s.allowed_origins_list == ["https://example.com"]


class TestProductionValidation:
    def test_production_wildcard_cors_warns(self):
        """Production + wildcard CORS emits a security warning (lines 89-91)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s = Settings(
                environment="production",
                allowed_origins="*",
                secret_key="prod-secret",
                _env_file="nonexistent.env",
            )
            security_warnings = [x for x in w if "SECURITY WARNING" in str(x.message)]
            assert len(security_warnings) == 1
            assert "CORS allows all origins" in str(security_warnings[0].message)
            assert s.is_production is True

    def test_production_specific_origins_no_warning(self):
        """Production with specific origins does not warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Settings(
                environment="production",
                allowed_origins="https://example.com",
                secret_key="prod-secret",
                _env_file="nonexistent.env",
            )
            security_warnings = [x for x in w if "SECURITY WARNING" in str(x.message)]
            assert len(security_warnings) == 0

    def test_development_wildcard_no_warning(self):
        """Development mode with wildcard does not warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Settings(
                environment="development",
                allowed_origins="*",
                _env_file="nonexistent.env",
            )
            security_warnings = [x for x in w if "SECURITY WARNING" in str(x.message)]
            assert len(security_warnings) == 0
