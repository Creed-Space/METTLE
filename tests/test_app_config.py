"""Tests for mettle/app_config.py — Pydantic Settings."""

from mettle.app_config import MettleSettings


class TestMettleSettings:
    def test_default_settings(self, monkeypatch, tmp_path):
        """Default settings have expected values when no env vars or .env file."""
        # Clear any env vars set by conftest or other tests
        for key in [
            "METTLE_DEV_MODE",
            "METTLE_REDIS_URL",
            "METTLE_API_KEYS",
            "METTLE_CORS_ORIGINS",
            "METTLE_VCP_SIGNING_KEY",
            "METTLE_SECRET_KEY",
            "METTLE_ADMIN_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)
        # Point env_file to a nonexistent path so pydantic-settings won't read .env
        s = MettleSettings(_env_file=str(tmp_path / "nonexistent.env"))
        assert s.dev_mode is False
        assert s.redis_url == "redis://localhost:6379"
        assert s.vcp_signing_key == ""
        assert s.api_keys == ""
        assert s.cors_origins == "*"

    def test_env_prefix(self):
        """Settings use METTLE_ env prefix."""
        assert MettleSettings.model_config["env_prefix"] == "METTLE_"

    def test_custom_values(self, monkeypatch):
        """Settings can be overridden via env vars."""
        monkeypatch.setenv("METTLE_DEV_MODE", "true")
        monkeypatch.setenv("METTLE_REDIS_URL", "redis://custom:1234")
        monkeypatch.setenv("METTLE_API_KEYS", "k1,k2")
        monkeypatch.setenv("METTLE_CORS_ORIGINS", "http://localhost:3000")
        monkeypatch.setenv("METTLE_VCP_SIGNING_KEY", "test-key")
        s = MettleSettings()
        assert s.dev_mode is True
        assert s.redis_url == "redis://custom:1234"
        assert s.api_keys == "k1,k2"
        assert s.cors_origins == "http://localhost:3000"
        assert s.vcp_signing_key == "test-key"
