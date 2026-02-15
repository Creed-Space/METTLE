"""Integration tests for METTLE v2 API.

Tests the API endpoints end-to-end with FakeRedis, focusing on:
- Authentication enforcement (403 on unauthenticated, per FastAPI HTTPBearer default)
- Ownership enforcement (403 on wrong user)
- Full session lifecycle (create -> verify -> result)
- Correct answer submission and passing verification
- Multi-round novel reasoning flow
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mettle.auth import require_authenticated_user
from mettle.router import get_session_manager, router
from mettle.challenge_adapter import ChallengeAdapter
from mettle.api_models import SUITE_NAMES
from mettle.session_manager import SessionManager

# ---- Shared FakeRedis (same as unit tests) ----


class FakeRedis:
    """In-memory async Redis mock for integration testing."""

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


# ---- Fixtures ----


def _make_mock_user(user_id: str = "test-user-123") -> MagicMock:
    user = MagicMock()
    user.user_id = user_id
    user.session_id = "test-session-abc"
    return user


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def mock_user() -> MagicMock:
    return _make_mock_user()


@pytest.fixture
def app_authenticated(mock_user: MagicMock, fake_redis: FakeRedis) -> FastAPI:
    """App with auth override (authenticated user)."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_authenticated_user] = lambda: mock_user

    async def mock_get_manager() -> SessionManager:
        return SessionManager(fake_redis)

    app.dependency_overrides[get_session_manager] = mock_get_manager
    return app


@pytest.fixture
def client(app_authenticated: FastAPI) -> TestClient:
    return TestClient(app_authenticated)


@pytest.fixture
def app_no_auth(fake_redis: FakeRedis) -> FastAPI:
    """App WITHOUT auth override -- tests 401 behavior."""
    app = FastAPI()
    app.include_router(router)

    async def mock_get_manager() -> SessionManager:
        return SessionManager(fake_redis)

    app.dependency_overrides[get_session_manager] = mock_get_manager
    return app


@pytest.fixture
def unauthenticated_client(app_no_auth: FastAPI) -> TestClient:
    return TestClient(app_no_auth)


@pytest.fixture
def wrong_user_client(fake_redis: FakeRedis) -> TestClient:
    """Client authenticated as a DIFFERENT user (for 403 tests)."""
    app = FastAPI()
    app.include_router(router)
    wrong_user = _make_mock_user(user_id="wrong-user-456")
    app.dependency_overrides[require_authenticated_user] = lambda: wrong_user

    async def mock_get_manager() -> SessionManager:
        return SessionManager(fake_redis)

    app.dependency_overrides[get_session_manager] = mock_get_manager
    return TestClient(app)


# ---- Authentication Enforcement (401) ----


class TestAuthEnforcement:
    """Verify all endpoints reject unauthenticated requests.

    FastAPI's HTTPBearer returns 401 when no credentials are provided.
    The auth module also returns 401 when credentials are provided but invalid.
    """

    def test_list_suites_requires_auth(self, unauthenticated_client: TestClient) -> None:
        resp = unauthenticated_client.get("/api/mettle/suites")
        assert resp.status_code in (401, 403)

    def test_get_suite_requires_auth(self, unauthenticated_client: TestClient) -> None:
        resp = unauthenticated_client.get("/api/mettle/suites/adversarial")
        assert resp.status_code in (401, 403)

    def test_create_session_requires_auth(self, unauthenticated_client: TestClient) -> None:
        resp = unauthenticated_client.post(
            "/api/mettle/sessions",
            json={"suites": ["adversarial"]},
        )
        assert resp.status_code in (401, 403)

    def test_get_session_requires_auth(self, unauthenticated_client: TestClient) -> None:
        resp = unauthenticated_client.get("/api/mettle/sessions/some-id")
        assert resp.status_code in (401, 403)

    def test_cancel_session_requires_auth(self, unauthenticated_client: TestClient) -> None:
        resp = unauthenticated_client.delete("/api/mettle/sessions/some-id")
        assert resp.status_code in (401, 403)

    def test_verify_requires_auth(self, unauthenticated_client: TestClient) -> None:
        resp = unauthenticated_client.post(
            "/api/mettle/sessions/some-id/verify",
            json={"suite": "adversarial", "answers": {}},
        )
        assert resp.status_code in (401, 403)

    def test_submit_round_requires_auth(self, unauthenticated_client: TestClient) -> None:
        resp = unauthenticated_client.post(
            "/api/mettle/sessions/some-id/rounds/1/answer",
            json={"answers": {}},
        )
        assert resp.status_code in (401, 403)

    def test_get_result_requires_auth(self, unauthenticated_client: TestClient) -> None:
        resp = unauthenticated_client.get("/api/mettle/sessions/some-id/result")
        assert resp.status_code in (401, 403)


# ---- Ownership Enforcement (403) ----


class TestOwnershipEnforcement:
    """Verify endpoints return 403 when accessed by wrong user."""

    def test_get_session_wrong_owner(self, client: TestClient, wrong_user_client: TestClient) -> None:
        # Create as correct user
        resp = client.post("/api/mettle/sessions", json={"suites": ["adversarial"]})
        session_id = resp.json()["session_id"]

        # Access as wrong user -- 403
        resp = wrong_user_client.get(f"/api/mettle/sessions/{session_id}")
        assert resp.status_code == 403

    def test_cancel_session_wrong_owner(self, client: TestClient, wrong_user_client: TestClient) -> None:
        resp = client.post("/api/mettle/sessions", json={"suites": ["adversarial"]})
        session_id = resp.json()["session_id"]

        resp = wrong_user_client.delete(f"/api/mettle/sessions/{session_id}")
        assert resp.status_code == 404  # cancel returns 404 for wrong user (security: don't reveal existence)

    def test_verify_wrong_owner(self, client: TestClient, wrong_user_client: TestClient) -> None:
        resp = client.post("/api/mettle/sessions", json={"suites": ["adversarial"]})
        session_id = resp.json()["session_id"]

        resp = wrong_user_client.post(
            f"/api/mettle/sessions/{session_id}/verify",
            json={"suite": "adversarial", "answers": {}},
        )
        assert resp.status_code == 403

    def test_get_result_wrong_owner(self, client: TestClient, wrong_user_client: TestClient) -> None:
        resp = client.post("/api/mettle/sessions", json={"suites": ["adversarial"]})
        session_id = resp.json()["session_id"]

        resp = wrong_user_client.get(f"/api/mettle/sessions/{session_id}/result")
        assert resp.status_code == 403


# ---- Full Session Lifecycle ----


class TestFullSessionLifecycle:
    """End-to-end session flows: create -> verify -> result."""

    def test_single_suite_lifecycle(self, client: TestClient) -> None:
        """Create session, verify suite, get result."""
        # 1. Create
        resp = client.post("/api/mettle/sessions", json={"suites": ["adversarial"]})
        assert resp.status_code == 201
        data = resp.json()
        session_id = data["session_id"]
        assert "adversarial" in data["challenges"]
        assert data["time_budget_ms"] > 0

        # 2. Check status
        resp = client.get(f"/api/mettle/sessions/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "challenges_generated"

        # 3. Verify
        resp = client.post(
            f"/api/mettle/sessions/{session_id}/verify",
            json={"suite": "adversarial", "answers": {"dynamic_math": {"computed": 0, "time_ms": 50}}},
        )
        assert resp.status_code == 200

        # 4. Get result
        resp = client.get(f"/api/mettle/sessions/{session_id}/result")
        assert resp.status_code == 200
        result = resp.json()
        assert result["status"] == "completed"
        assert "adversarial" in result["suites_completed"]

    def test_multi_suite_lifecycle(self, client: TestClient) -> None:
        """Create with 3 suites, verify each, get combined result."""
        suites = ["adversarial", "native", "self-reference"]
        resp = client.post("/api/mettle/sessions", json={"suites": suites})
        assert resp.status_code == 201
        session_id = resp.json()["session_id"]

        # Verify each suite
        for suite in suites:
            resp = client.post(
                f"/api/mettle/sessions/{session_id}/verify",
                json={"suite": suite, "answers": {}},
            )
            assert resp.status_code == 200

        # Get combined result
        resp = client.get(f"/api/mettle/sessions/{session_id}/result")
        assert resp.status_code == 200
        result = resp.json()
        assert result["status"] == "completed"
        assert set(result["suites_completed"]) == set(suites)

    def test_cancel_then_verify_fails(self, client: TestClient) -> None:
        """Cancel session, then verify should fail."""
        resp = client.post("/api/mettle/sessions", json={"suites": ["adversarial"]})
        session_id = resp.json()["session_id"]

        # Cancel
        resp = client.delete(f"/api/mettle/sessions/{session_id}")
        assert resp.status_code == 204

        # Verify should fail
        resp = client.post(
            f"/api/mettle/sessions/{session_id}/verify",
            json={"suite": "adversarial", "answers": {}},
        )
        assert resp.status_code == 400


# ---- Correct Answer Verification ----


class TestCorrectAnswerVerification:
    """Verify that submitting correct answers produces passing results."""

    def test_adversarial_correct_answers_pass(self, client: TestClient) -> None:
        """Generate adversarial, peek at server answers, submit correct ones."""
        # Generate challenge pair directly
        client_data, server_answers = ChallengeAdapter.generate_adversarial()

        # Create session
        resp = client.post("/api/mettle/sessions", json={"suites": ["adversarial"]})
        assert resp.status_code == 201
        resp.json()["session_id"]

        # Build correct answers from server data
        correct_answers: dict[str, Any] = {
            "dynamic_math": {
                "computed": server_answers["dynamic_math"]["expected"],
                "time_ms": 10,
            },
            "chained_reasoning": {
                "computed_final": server_answers["chained_reasoning"]["expected_final"],
            },
            "time_locked_secret": {
                "recalled": server_answers["time_locked_secret"]["secret"],
            },
        }

        # We can't inject these answers into the session (the session generated its own),
        # but we CAN verify the evaluation logic directly
        result = ChallengeAdapter.evaluate_single_shot("adversarial", correct_answers, server_answers)
        assert result["passed"] is True
        assert result["score"] == 1.0

    def test_native_correct_answers_pass(self) -> None:
        """Verify native suite evaluation with correct answers."""
        client_data, server_answers = ChallengeAdapter.generate_native()
        target = server_answers["batch_coherence"]["target"]

        correct_answers: dict[str, Any] = {
            "batch_coherence": {
                "responses": [f"{c}ordial" for c in target],
            },
            "calibrated_uncertainty": {
                "confidences": server_answers["calibrated_uncertainty"]["ground_truth"],
            },
        }

        result = ChallengeAdapter.evaluate_single_shot("native", correct_answers, server_answers)
        assert result["passed"] is True
        assert result["score"] == 1.0


# ---- Multi-Round Novel Reasoning ----


class TestNovelReasoningFlow:
    """Test multi-round novel reasoning (Suite 10) flow."""

    def test_easy_novel_reasoning_lifecycle(self, client: TestClient) -> None:
        """Create session with novel-reasoning, submit all rounds, get result."""
        resp = client.post(
            "/api/mettle/sessions",
            json={"suites": ["novel-reasoning"], "difficulty": "easy"},
        )
        assert resp.status_code == 201
        data = resp.json()
        session_id = data["session_id"]
        assert "novel-reasoning" in data["challenges"]

        challenges = data["challenges"]["novel-reasoning"]["challenges"]
        num_rounds = data["challenges"]["novel-reasoning"]["num_rounds"]

        # Submit each round
        for round_num in range(1, num_rounds + 1):
            resp = client.post(
                f"/api/mettle/sessions/{session_id}/rounds/{round_num}/answer",
                json={"answers": {name: {} for name in challenges}},
            )
            assert resp.status_code == 200
            round_data = resp.json()
            assert round_data["round_num"] == round_num
            assert "time_remaining_ms" in round_data

            if round_num < num_rounds:
                # Non-final rounds may have next_round_data
                pass

        # Get final result
        resp = client.get(f"/api/mettle/sessions/{session_id}/result")
        assert resp.status_code == 200
        result = resp.json()
        assert result["status"] == "completed"
        assert "novel-reasoning" in result["suites_completed"]

    def test_novel_reasoning_round_feedback(self, client: TestClient) -> None:
        """Submit round, then retrieve feedback separately."""
        resp = client.post(
            "/api/mettle/sessions",
            json={"suites": ["novel-reasoning"], "difficulty": "easy"},
        )
        session_id = resp.json()["session_id"]

        # Submit round 1
        resp = client.post(
            f"/api/mettle/sessions/{session_id}/rounds/1/answer",
            json={"answers": {}},
        )
        assert resp.status_code == 200

        # Get round feedback
        resp = client.get(f"/api/mettle/sessions/{session_id}/rounds/1/feedback")
        assert resp.status_code == 200
        feedback = resp.json()
        assert feedback["round"] == 1
        assert "accuracy" in feedback


# ---- All Suites Smoke Test ----


class TestAllSuitesSmoke:
    """Verify all 10 suites can be created and verified."""

    def test_create_all_suites(self, client: TestClient) -> None:
        """Create session with all suites."""
        resp = client.post("/api/mettle/sessions", json={"suites": ["all"]})
        assert resp.status_code == 201
        data = resp.json()
        assert len(data["suites"]) == 10
        assert set(data["suites"]) == set(SUITE_NAMES)

    def test_suite_info_all_suites(self, client: TestClient) -> None:
        """Verify suite info is available for all suites."""
        resp = client.get("/api/mettle/suites")
        assert resp.status_code == 200
        suites = resp.json()
        assert len(suites) == 10
        names = {s["name"] for s in suites}
        assert names == set(SUITE_NAMES)

    def test_each_single_shot_suite_verifiable(self, client: TestClient) -> None:
        """Each single-shot suite can create + verify independently."""
        single_shot = [s for s in SUITE_NAMES if s != "novel-reasoning"]
        for suite_name in single_shot:
            resp = client.post("/api/mettle/sessions", json={"suites": [suite_name]})
            assert resp.status_code == 201, f"Failed to create session for {suite_name}"
            session_id = resp.json()["session_id"]

            resp = client.post(
                f"/api/mettle/sessions/{session_id}/verify",
                json={"suite": suite_name, "answers": {}},
            )
            assert resp.status_code == 200, f"Failed to verify {suite_name}"
            data = resp.json()
            assert data["suite"] == suite_name
            assert "passed" in data
            assert "score" in data


# ---- Error Handling ----


class TestErrorHandling:
    """Verify proper error responses for invalid inputs."""

    def test_verify_nonexistent_session(self, client: TestClient) -> None:
        resp = client.post(
            "/api/mettle/sessions/nonexistent/verify",
            json={"suite": "adversarial", "answers": {}},
        )
        assert resp.status_code == 404

    def test_submit_round_nonexistent_session(self, client: TestClient) -> None:
        resp = client.post(
            "/api/mettle/sessions/nonexistent/rounds/1/answer",
            json={"answers": {}},
        )
        assert resp.status_code == 404

    def test_duplicate_suite_verification(self, client: TestClient) -> None:
        """Verifying the same suite twice should fail."""
        resp = client.post("/api/mettle/sessions", json={"suites": ["adversarial"]})
        session_id = resp.json()["session_id"]

        # First verify
        resp = client.post(
            f"/api/mettle/sessions/{session_id}/verify",
            json={"suite": "adversarial", "answers": {}},
        )
        assert resp.status_code == 200

        # Second verify -- already completed, should fail
        # Session is now completed (single suite), so verification should fail
        resp = client.post(
            f"/api/mettle/sessions/{session_id}/verify",
            json={"suite": "adversarial", "answers": {}},
        )
        assert resp.status_code == 400

    def test_novel_reasoning_wrong_round_order(self, client: TestClient) -> None:
        """Submitting round 2 before round 1 should fail."""
        resp = client.post(
            "/api/mettle/sessions",
            json={"suites": ["novel-reasoning"], "difficulty": "easy"},
        )
        session_id = resp.json()["session_id"]

        resp = client.post(
            f"/api/mettle/sessions/{session_id}/rounds/2/answer",
            json={"answers": {}},
        )
        assert resp.status_code == 400

    def test_invalid_difficulty(self, client: TestClient) -> None:
        """Invalid difficulty level should fail validation."""
        resp = client.post(
            "/api/mettle/sessions",
            json={"suites": ["adversarial"], "difficulty": "impossible"},
        )
        assert resp.status_code == 422  # Pydantic validation error
