"""Tests for mettle/auth.py — API key bearer authentication."""

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from mettle.auth import AuthenticatedUser, require_authenticated_user


class TestRequireAuthenticatedUser:
    @pytest.mark.asyncio
    async def test_valid_api_key(self, monkeypatch):
        """Valid API key returns AuthenticatedUser."""
        monkeypatch.setenv("METTLE_API_KEYS", "key1,key2")
        monkeypatch.setenv("METTLE_DEV_MODE", "false")
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="key1")
        result = await require_authenticated_user(creds)
        assert isinstance(result, AuthenticatedUser)
        assert result.user_id == "key:key1..."

    @pytest.mark.asyncio
    async def test_second_valid_key(self, monkeypatch):
        """Second key in comma-separated list also works."""
        monkeypatch.setenv("METTLE_API_KEYS", "key1,key2")
        monkeypatch.setenv("METTLE_DEV_MODE", "false")
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="key2")
        result = await require_authenticated_user(creds)
        assert result.user_id == "key:key2..."

    @pytest.mark.asyncio
    async def test_invalid_key_raises_401(self, monkeypatch):
        """Invalid API key raises HTTPException with 401."""
        monkeypatch.setenv("METTLE_API_KEYS", "key1")
        monkeypatch.setenv("METTLE_DEV_MODE", "false")
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="badkey")
        with pytest.raises(HTTPException) as exc_info:
            await require_authenticated_user(creds)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid API key"

    @pytest.mark.asyncio
    async def test_dev_mode_bypass(self, monkeypatch):
        """Dev mode allows any key through."""
        monkeypatch.setenv("METTLE_API_KEYS", "")
        monkeypatch.setenv("METTLE_DEV_MODE", "true")
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="anything")
        result = await require_authenticated_user(creds)
        assert isinstance(result, AuthenticatedUser)
        assert result.user_id == "key:anything..."

    @pytest.mark.asyncio
    async def test_empty_api_keys_rejects(self, monkeypatch):
        """Empty METTLE_API_KEYS with dev mode off rejects all keys."""
        monkeypatch.setenv("METTLE_API_KEYS", "")
        monkeypatch.setenv("METTLE_DEV_MODE", "false")
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="anykey")
        with pytest.raises(HTTPException) as exc_info:
            await require_authenticated_user(creds)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_no_env_vars_rejects(self, monkeypatch):
        """Missing env vars default to rejecting."""
        monkeypatch.delenv("METTLE_API_KEYS", raising=False)
        monkeypatch.delenv("METTLE_DEV_MODE", raising=False)
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="anykey")
        with pytest.raises(HTTPException) as exc_info:
            await require_authenticated_user(creds)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_user_id_truncation(self, monkeypatch):
        """User ID truncates key to first 8 chars."""
        monkeypatch.setenv("METTLE_API_KEYS", "abcdefghijklmnop")
        monkeypatch.setenv("METTLE_DEV_MODE", "false")
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="abcdefghijklmnop")
        result = await require_authenticated_user(creds)
        assert result.user_id == "key:abcdefgh..."


class TestAuthenticatedUserModel:
    def test_model_fields(self):
        """AuthenticatedUser has expected fields."""
        user = AuthenticatedUser(user_id="test-user")
        assert user.user_id == "test-user"
