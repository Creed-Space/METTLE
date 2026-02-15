"""Tests for mettle.auth module - API key bearer authentication (covers lines 21-26)."""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mettle.auth import AuthenticatedUser, require_authenticated_user

# Minimal FastAPI app for testing the dependency
_test_app = FastAPI()


@_test_app.get("/protected")
async def protected_route(user: AuthenticatedUser = pytest.importorskip("fastapi").Depends(require_authenticated_user)):
    return {"user_id": user.user_id}


@pytest.fixture
def auth_client():
    return TestClient(_test_app)


class TestAuthenticatedUser:
    """Test AuthenticatedUser model."""

    def test_create(self):
        u = AuthenticatedUser(user_id="key:abc12345...")
        assert u.user_id == "key:abc12345..."


class TestRequireAuthenticatedUser:
    """Test require_authenticated_user dependency via TestClient."""

    def test_valid_api_key(self, auth_client):
        with patch.dict("os.environ", {"METTLE_DEV_MODE": "false", "METTLE_API_KEYS": "my-secret-key,other-key"}):
            resp = auth_client.get("/protected", headers={"Authorization": "Bearer my-secret-key"})
        assert resp.status_code == 200
        assert resp.json()["user_id"] == "key:my-secre..."

    def test_dev_mode_accepts_any_key(self, auth_client):
        with patch.dict("os.environ", {"METTLE_DEV_MODE": "true", "METTLE_API_KEYS": ""}):
            resp = auth_client.get("/protected", headers={"Authorization": "Bearer anything-goes"})
        assert resp.status_code == 200
        assert resp.json()["user_id"] == "key:anything..."

    def test_invalid_key_returns_401(self, auth_client):
        with patch.dict("os.environ", {"METTLE_DEV_MODE": "false", "METTLE_API_KEYS": "valid-key-only"}):
            resp = auth_client.get("/protected", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid API key"

    def test_missing_auth_header_returns_401(self, auth_client):
        """No Authorization header -> FastAPI's HTTPBearer returns 401."""
        resp = auth_client.get("/protected")
        assert resp.status_code == 401

    def test_empty_api_keys_env(self, auth_client):
        """When METTLE_API_KEYS is empty string, split produces [''], key won't match."""
        with patch.dict("os.environ", {"METTLE_DEV_MODE": "false", "METTLE_API_KEYS": ""}):
            resp = auth_client.get("/protected", headers={"Authorization": "Bearer some-key"})
        assert resp.status_code == 401

    def test_dev_mode_case_insensitive(self, auth_client):
        with patch.dict("os.environ", {"METTLE_DEV_MODE": "True", "METTLE_API_KEYS": ""}):
            resp = auth_client.get("/protected", headers={"Authorization": "Bearer x"})
        assert resp.status_code == 200

    def test_user_id_truncation(self, auth_client):
        """user_id uses first 8 chars of the key."""
        with patch.dict("os.environ", {"METTLE_DEV_MODE": "false", "METTLE_API_KEYS": "abcdefghijklmnop"}):
            resp = auth_client.get("/protected", headers={"Authorization": "Bearer abcdefghijklmnop"})
        assert resp.status_code == 200
        assert resp.json()["user_id"] == "key:abcdefgh..."
