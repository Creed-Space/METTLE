"""Tests for remaining coverage gaps in mettle/router.py.

Covers:
- get_session_manager when Redis unavailable
- get_session_status with different user and elapsed time
- cancel_session with not-found session
- verify_single_shot ownership, not-found, and ValueError
- submit_round_answer ownership, not-found, and ValueError
- get_round_feedback ownership, not-found, and round-not-completed
- get_session_result ownership, not-found, and not-completed

Uses the same FakeRedis pattern from test_mettle_api.py with
FastAPI TestClient and dependency overrides.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mettle.session_manager import SessionManager

# ---------------------------------------------------------------------------
# FakeRedis (same pattern as test_mettle_api.py)
# ---------------------------------------------------------------------------


class FakeRedis:
    """In-memory async Redis mock for testing."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._sets: dict[str, set[str]] = {}
        self._ttls: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value
        self._ttls[key] = ttl

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def sadd(self, key: str, *values: str) -> int:
        if key not in self._sets:
            self._sets[key] = set()
        self._sets[key].update(values)
        return len(values)

    async def srem(self, key: str, *values: str) -> int:
        if key not in self._sets:
            return 0
        before = len(self._sets[key])
        self._sets[key] -= set(values)
        return before - len(self._sets[key])

    async def scard(self, key: str) -> int:
        return len(self._sets.get(key, set()))

    async def incr(self, key: str) -> int:
        val = int(self._store.get(key, "0"))
        val += 1
        self._store[key] = str(val)
        return val

    async def expire(self, key: str, ttl: int) -> None:
        self._ttls[key] = ttl

    def pipeline(self) -> FakeRedisPipeline:
        return FakeRedisPipeline(self)


class FakeRedisPipeline:
    """Accumulates commands then executes them."""

    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._commands: list[tuple[str, tuple[Any, ...]]] = []

    def setex(self, key: str, ttl: int, value: str) -> FakeRedisPipeline:
        self._commands.append(("setex", (key, ttl, value)))
        return self

    def sadd(self, key: str, *values: str) -> FakeRedisPipeline:
        self._commands.append(("sadd", (key, *values)))
        return self

    def srem(self, key: str, *values: str) -> FakeRedisPipeline:
        self._commands.append(("srem", (key, *values)))
        return self

    def incr(self, key: str) -> FakeRedisPipeline:
        self._commands.append(("incr", (key,)))
        return self

    def expire(self, key: str, ttl: int) -> FakeRedisPipeline:
        self._commands.append(("expire", (key, ttl)))
        return self

    async def execute(self) -> list[Any]:
        results = []
        for cmd, args in self._commands:
            fn = getattr(self._redis, cmd)
            result = await fn(*args)
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def session_manager(fake_redis: FakeRedis) -> SessionManager:
    return SessionManager(fake_redis)


def _make_mock_user(user_id: str = "test-user-123") -> MagicMock:
    user = MagicMock()
    user.user_id = user_id
    user.session_id = "test-session-abc"
    return user


@pytest.fixture
def mock_user() -> MagicMock:
    return _make_mock_user()


@pytest.fixture
def mock_user_other() -> MagicMock:
    return _make_mock_user("other-user-999")


@pytest.fixture
def app(mock_user: MagicMock, fake_redis: FakeRedis) -> FastAPI:
    """Test FastAPI app with mocked dependencies."""
    from mettle.auth import require_authenticated_user
    from mettle.router import get_session_manager, router

    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[require_authenticated_user] = lambda: mock_user

    async def mock_get_manager() -> SessionManager:
        return SessionManager(fake_redis)

    app.dependency_overrides[get_session_manager] = mock_get_manager
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _app_with_user(user: MagicMock, fake_redis: FakeRedis) -> FastAPI:
    """Build a test app with a specific user mock."""
    from mettle.auth import require_authenticated_user
    from mettle.router import get_session_manager, router

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_authenticated_user] = lambda: user

    async def mock_get_manager() -> SessionManager:
        return SessionManager(fake_redis)

    app.dependency_overrides[get_session_manager] = mock_get_manager
    return app


# ---------------------------------------------------------------------------
# get_session_manager: Redis unavailable
# ---------------------------------------------------------------------------


class TestGetSessionManager:
    """Test get_session_manager dependency."""

    @pytest.mark.asyncio
    async def test_redis_unavailable_raises_503(self) -> None:
        from mettle.router import get_session_manager

        with patch("mettle.router.get_redis_client", new_callable=AsyncMock) as mock_redis:
            mock_redis.return_value = None
            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc_info:
                await get_session_manager()
            assert exc_info.value.status_code == 503
            assert "Redis" in exc_info.value.detail


# ---------------------------------------------------------------------------
# get_session_status: different user and elapsed time
# ---------------------------------------------------------------------------


class TestGetSessionStatus:
    """Test get_session_status endpoint edge cases."""

    def test_different_user_returns_403(self, fake_redis: FakeRedis) -> None:
        owner = _make_mock_user("owner-user")
        other = _make_mock_user("other-user")

        # Create session as owner
        app_owner = _app_with_user(owner, fake_redis)
        owner_client = TestClient(app_owner)

        create_resp = owner_client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["adversarial"]},
        )
        assert create_resp.status_code == 201
        session_id = create_resp.json()["session_id"]

        # Try to access as different user
        app_other = _app_with_user(other, fake_redis)
        other_client = TestClient(app_other)

        resp = other_client.get(f"/api/v1/mettle/sessions/{session_id}")
        assert resp.status_code == 403

    def test_session_with_start_time_has_elapsed_ms(self, fake_redis: FakeRedis) -> None:
        user = _make_mock_user("user-elapsed")
        app = _app_with_user(user, fake_redis)
        test_client = TestClient(app)

        # Create session
        create_resp = test_client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["adversarial"]},
        )
        session_id = create_resp.json()["session_id"]

        # Manually set start_time in the stored session data by
        # directly mutating the FakeRedis in-memory store
        key = f"mettle:session:{session_id}"
        stored = fake_redis._store.get(key)
        assert stored is not None
        session_data = json.loads(stored)
        session_data["start_time"] = time.time() - 2.0  # 2 seconds ago
        fake_redis._store[key] = json.dumps(session_data)

        resp = test_client.get(f"/api/v1/mettle/sessions/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        # elapsed_ms should be at least 1000 (2s = 2000ms, with some tolerance)
        assert data["elapsed_ms"] >= 1000


# ---------------------------------------------------------------------------
# cancel_session: not found
# ---------------------------------------------------------------------------


class TestCancelSession:
    """Test cancel_session endpoint edge cases."""

    def test_session_not_found_returns_404(self, client: TestClient) -> None:
        resp = client.delete("/api/v1/mettle/sessions/nonexistent-session-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# verify_single_shot: not found, different user, ValueError
# ---------------------------------------------------------------------------


class TestVerifySingleShot:
    """Test verify_single_shot endpoint edge cases."""

    def test_session_not_found_returns_404(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/mettle/sessions/nonexistent-id/verify",
            json={"suite": "adversarial", "answers": {}},
        )
        assert resp.status_code == 404

    def test_different_user_returns_403(self, fake_redis: FakeRedis) -> None:
        owner = _make_mock_user("owner-user")
        other = _make_mock_user("other-user")

        app_owner = _app_with_user(owner, fake_redis)
        owner_client = TestClient(app_owner)

        create_resp = owner_client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["adversarial"]},
        )
        session_id = create_resp.json()["session_id"]

        app_other = _app_with_user(other, fake_redis)
        other_client = TestClient(app_other)

        resp = other_client.post(
            f"/api/v1/mettle/sessions/{session_id}/verify",
            json={"suite": "adversarial", "answers": {}},
        )
        assert resp.status_code == 403

    def test_value_error_from_manager_returns_400(self, client: TestClient) -> None:
        # Create session with adversarial, then try to verify 'native' (not in session)
        create_resp = client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["adversarial"]},
        )
        session_id = create_resp.json()["session_id"]

        resp = client.post(
            f"/api/v1/mettle/sessions/{session_id}/verify",
            json={"suite": "native", "answers": {}},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# submit_round_answer: not found, different user, ValueError
# ---------------------------------------------------------------------------


class TestSubmitRoundAnswer:
    """Test submit_round_answer endpoint edge cases."""

    def test_session_not_found_returns_404(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/mettle/sessions/nonexistent-id/rounds/1/answer",
            json={"answers": {}},
        )
        assert resp.status_code == 404

    def test_different_user_returns_403(self, fake_redis: FakeRedis) -> None:
        owner = _make_mock_user("owner-user")
        other = _make_mock_user("other-user")

        app_owner = _app_with_user(owner, fake_redis)
        owner_client = TestClient(app_owner)

        create_resp = owner_client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["novel-reasoning"], "difficulty": "easy"},
        )
        session_id = create_resp.json()["session_id"]

        app_other = _app_with_user(other, fake_redis)
        other_client = TestClient(app_other)

        resp = other_client.post(
            f"/api/v1/mettle/sessions/{session_id}/rounds/1/answer",
            json={"answers": {}},
        )
        assert resp.status_code == 403

    def test_value_error_from_manager_returns_400(self, client: TestClient) -> None:
        # Create session with novel-reasoning, then submit wrong round order
        create_resp = client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["novel-reasoning"], "difficulty": "easy"},
        )
        session_id = create_resp.json()["session_id"]

        # Submit round 2 before round 1
        resp = client.post(
            f"/api/v1/mettle/sessions/{session_id}/rounds/2/answer",
            json={"answers": {}},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# get_round_feedback: not found, different user, round not completed
# ---------------------------------------------------------------------------


class TestGetRoundFeedback:
    """Test get_round_feedback endpoint edge cases."""

    def test_session_not_found_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/v1/mettle/sessions/nonexistent-id/rounds/1/feedback")
        assert resp.status_code == 404

    def test_different_user_returns_403(self, fake_redis: FakeRedis) -> None:
        owner = _make_mock_user("owner-user")
        other = _make_mock_user("other-user")

        app_owner = _app_with_user(owner, fake_redis)
        owner_client = TestClient(app_owner)

        create_resp = owner_client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["novel-reasoning"], "difficulty": "easy"},
        )
        session_id = create_resp.json()["session_id"]

        app_other = _app_with_user(other, fake_redis)
        other_client = TestClient(app_other)

        resp = other_client.get(f"/api/v1/mettle/sessions/{session_id}/rounds/1/feedback")
        assert resp.status_code == 403

    def test_round_not_yet_completed_returns_404(self, client: TestClient) -> None:
        create_resp = client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["novel-reasoning"], "difficulty": "easy"},
        )
        session_id = create_resp.json()["session_id"]

        # Round 1 has not been submitted yet
        resp = client.get(f"/api/v1/mettle/sessions/{session_id}/rounds/1/feedback")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# get_session_result: not found, different user, not completed
# ---------------------------------------------------------------------------


class TestGetSessionResult:
    """Test get_session_result endpoint edge cases."""

    def test_session_not_found_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/v1/mettle/sessions/nonexistent-id/result")
        assert resp.status_code == 404

    def test_different_user_returns_403(self, fake_redis: FakeRedis) -> None:
        owner = _make_mock_user("owner-user")
        other = _make_mock_user("other-user")

        app_owner = _app_with_user(owner, fake_redis)
        owner_client = TestClient(app_owner)

        create_resp = owner_client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["adversarial"]},
        )
        session_id = create_resp.json()["session_id"]

        app_other = _app_with_user(other, fake_redis)
        other_client = TestClient(app_other)

        resp = other_client.get(f"/api/v1/mettle/sessions/{session_id}/result")
        assert resp.status_code == 403

    def test_session_not_completed_returns_400(self, client: TestClient) -> None:
        create_resp = client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["adversarial"]},
        )
        session_id = create_resp.json()["session_id"]

        # Session is in challenges_generated state, not completed
        resp = client.get(f"/api/v1/mettle/sessions/{session_id}/result")
        assert resp.status_code == 400
