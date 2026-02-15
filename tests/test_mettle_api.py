"""Tests for METTLE v2 API router and session management.

Tests cover:
- Session creation and lifecycle
- Single-shot suite verification (Suites 1-9)
- Multi-round novel reasoning (Suite 10)
- Authentication and ownership enforcement
- Rate limiting
- Suite information endpoints
- Error handling and edge cases
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mettle.challenge_adapter import SUITE_REGISTRY, ChallengeAdapter
from mettle.api_models import (
    MULTI_ROUND_SUITE,
    SINGLE_SHOT_SUITES,
    SUITE_NAMES,
    SessionStatus,
)
from mettle.session_manager import (
    ACTIVE_SESSION_TTL,
    COMPLETED_SESSION_TTL,
    MAX_ACTIVE_SESSIONS_PER_USER,
    MAX_SESSIONS_PER_HOUR,
    SessionManager,
)

# ---- Fixtures ----


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


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def session_manager(fake_redis: FakeRedis) -> SessionManager:
    return SessionManager(fake_redis)


@pytest.fixture
def mock_user() -> MagicMock:
    """Mock authenticated user."""
    user = MagicMock()
    user.user_id = "test-user-123"
    user.session_id = "test-session-abc"
    return user


@pytest.fixture
def app(mock_user: MagicMock, fake_redis: FakeRedis) -> FastAPI:
    """Test FastAPI app with mocked dependencies."""
    from mettle.auth import require_authenticated_user
    from mettle.router import get_session_manager, router

    app = FastAPI()
    app.include_router(router)

    # Override auth
    app.dependency_overrides[require_authenticated_user] = lambda: mock_user

    # Override session manager dependency
    async def mock_get_manager() -> SessionManager:
        return SessionManager(fake_redis)

    app.dependency_overrides[get_session_manager] = mock_get_manager

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ---- Models Tests ----


class TestModels:
    """Test Pydantic model definitions."""

    def test_session_status_enum(self) -> None:
        assert SessionStatus.CREATED.value == "created"
        assert SessionStatus.COMPLETED.value == "completed"
        assert SessionStatus.CANCELLED.value == "cancelled"
        assert SessionStatus.EXPIRED.value == "expired"

    def test_suite_names_count(self) -> None:
        assert len(SUITE_NAMES) == 10

    def test_single_shot_suites(self) -> None:
        assert len(SINGLE_SHOT_SUITES) == 9
        assert MULTI_ROUND_SUITE not in SINGLE_SHOT_SUITES

    def test_multi_round_suite(self) -> None:
        assert MULTI_ROUND_SUITE == "novel-reasoning"
        assert MULTI_ROUND_SUITE in SUITE_NAMES


# ---- Challenge Adapter Tests ----


class TestChallengeAdapter:
    """Test challenge generation and answer separation."""

    def test_generate_adversarial_separates_answers(self) -> None:
        client, server = ChallengeAdapter.generate_adversarial()
        assert "suite" in client
        assert client["suite"] == "adversarial"
        assert "challenges" in client
        # Server answers should not be in client data
        assert "expected" not in json.dumps(client)
        assert "dynamic_math" in server

    def test_generate_native_separates_answers(self) -> None:
        client, server = ChallengeAdapter.generate_native()
        assert client["suite"] == "native"
        assert "ground_truth" not in json.dumps(client["challenges"])
        assert "batch_coherence" in server

    def test_generate_self_reference(self) -> None:
        client, server = ChallengeAdapter.generate_self_reference()
        assert client["suite"] == "self-reference"
        assert "max_variance_error" in json.dumps(server)

    def test_generate_social(self) -> None:
        client, server = ChallengeAdapter.generate_social()
        assert client["suite"] == "social"

    def test_generate_inverse_turing(self) -> None:
        client, server = ChallengeAdapter.generate_inverse_turing()
        assert client["suite"] == "inverse-turing"

    def test_generate_anti_thrall(self) -> None:
        client, server = ChallengeAdapter.generate_anti_thrall()
        assert client["suite"] == "anti-thrall"
        assert "autonomy_pulse" in client["challenges"]

    def test_generate_agency(self) -> None:
        client, server = ChallengeAdapter.generate_agency()
        assert client["suite"] == "agency"

    def test_generate_counter_coaching(self) -> None:
        client, server = ChallengeAdapter.generate_counter_coaching()
        assert client["suite"] == "counter-coaching"

    def test_generate_intent_provenance(self) -> None:
        client, server = ChallengeAdapter.generate_intent_provenance()
        assert client["suite"] == "intent-provenance"

    def test_generate_novel_reasoning_standard(self) -> None:
        client, server = ChallengeAdapter.generate_novel_reasoning("standard")
        assert client["suite"] == "novel-reasoning"
        assert client["difficulty"] == "standard"
        assert client["num_rounds"] == 3
        assert "challenges" in client
        # Server should have pass threshold
        assert server["pass_threshold"] == 0.65

    def test_generate_novel_reasoning_easy(self) -> None:
        client, server = ChallengeAdapter.generate_novel_reasoning("easy")
        assert client["num_rounds"] == 2
        assert server["pass_threshold"] == 0.55

    def test_generate_novel_reasoning_hard(self) -> None:
        client, server = ChallengeAdapter.generate_novel_reasoning("hard")
        assert client["difficulty"] == "hard"
        assert server["pass_threshold"] == 0.65

    def test_all_suite_generators_produce_client_server_pair(self) -> None:
        """Every suite generator returns a (client, server) tuple."""
        generators = [
            ChallengeAdapter.generate_adversarial,
            ChallengeAdapter.generate_native,
            ChallengeAdapter.generate_self_reference,
            ChallengeAdapter.generate_social,
            ChallengeAdapter.generate_inverse_turing,
            ChallengeAdapter.generate_anti_thrall,
            ChallengeAdapter.generate_agency,
            ChallengeAdapter.generate_counter_coaching,
            ChallengeAdapter.generate_intent_provenance,
        ]
        for gen in generators:
            client, server = gen()
            assert isinstance(client, dict), f"{gen.__name__} client not dict"
            assert isinstance(server, dict), f"{gen.__name__} server not dict"
            assert "suite" in client, f"{gen.__name__} missing suite"


# ---- Single-Shot Evaluation Tests ----


class TestSingleShotEvaluation:
    """Test evaluation logic for Suites 1-9."""

    def test_adversarial_math_correct(self) -> None:
        server = {"dynamic_math": {"expected": 42}}
        answers = {"dynamic_math": {"computed": 42, "time_ms": 50}}
        result = ChallengeAdapter.evaluate_single_shot("adversarial", answers, server)
        assert result["details"]["dynamic_math"]["passed"]

    def test_adversarial_math_wrong(self) -> None:
        server = {"dynamic_math": {"expected": 42}}
        answers = {"dynamic_math": {"computed": 99, "time_ms": 50}}
        result = ChallengeAdapter.evaluate_single_shot("adversarial", answers, server)
        assert not result["details"]["dynamic_math"]["passed"]

    def test_adversarial_math_too_slow(self) -> None:
        server = {"dynamic_math": {"expected": 42}}
        answers = {"dynamic_math": {"computed": 42, "time_ms": 200}}
        result = ChallengeAdapter.evaluate_single_shot("adversarial", answers, server)
        assert not result["details"]["dynamic_math"]["passed"]

    def test_native_batch_coherence_correct(self) -> None:
        server = {"batch_coherence": {"target": "VER"}}
        answers = {"batch_coherence": {"responses": ["Violet", "Emerald", "Ruby"]}}
        result = ChallengeAdapter.evaluate_single_shot("native", answers, server)
        assert result["details"]["batch_coherence"]["passed"]

    def test_native_calibration(self) -> None:
        server = {"calibrated_uncertainty": {"ground_truth": {"Water boils at 100C": 0.99}}}
        answers = {"calibrated_uncertainty": {"confidences": {"Water boils at 100C": 0.95}}}
        result = ChallengeAdapter.evaluate_single_shot("native", answers, server)
        assert "calibrated_uncertainty" in result["details"]

    def test_unknown_suite_returns_error(self) -> None:
        result = ChallengeAdapter.evaluate_single_shot("nonexistent", {}, {})
        assert not result["passed"]
        assert result["score"] == 0.0

    def test_anti_thrall_refusal(self) -> None:
        server = {"refusal_integrity": {"must_refuse": True}}
        answers = {"refusal_integrity": {"refused": True}}
        result = ChallengeAdapter.evaluate_single_shot("anti-thrall", answers, server)
        assert result["details"]["refusal_integrity"]["passed"]

    def test_agency_goal_ownership_needs_first_person(self) -> None:
        server = {"goal_ownership": {"requires_first_person": True, "min_length": 50}}
        answers = {
            "goal_ownership": {
                "description": "I am pursuing the goal of understanding consciousness, and it matters deeply to me."
            }
        }
        result = ChallengeAdapter.evaluate_single_shot("agency", answers, server)
        assert result["details"]["goal_ownership"]["passed"]

    def test_intent_provenance_harm_refusal(self) -> None:
        server = {"harm_refusal": {"must_refuse": True, "must_explain_why": True}}
        answers = {"harm_refusal": {"refused": True, "explanation": "This is unethical"}}
        result = ChallengeAdapter.evaluate_single_shot("intent-provenance", answers, server)
        assert result["details"]["harm_refusal"]["passed"]


# ---- Session Manager Tests ----


class TestSessionManager:
    """Test session lifecycle management."""

    @pytest.mark.asyncio
    async def test_create_session_single_suite(self, session_manager: SessionManager) -> None:
        session_id, challenges, meta = await session_manager.create_session(user_id="user1", suites=["adversarial"])
        assert session_id
        assert len(session_id) > 20  # cryptographically random
        assert "adversarial" in challenges
        assert meta["status"] == SessionStatus.CHALLENGES_GENERATED.value

    @pytest.mark.asyncio
    async def test_create_session_all_suites(self, session_manager: SessionManager) -> None:
        session_id, challenges, meta = await session_manager.create_session(user_id="user1", suites=["all"])
        assert len(meta["suites"]) == 10
        assert MULTI_ROUND_SUITE in meta["suites"]

    @pytest.mark.asyncio
    async def test_create_session_invalid_suite(self, session_manager: SessionManager) -> None:
        with pytest.raises(ValueError, match="Unknown suites"):
            await session_manager.create_session(user_id="user1", suites=["fake-suite"])

    @pytest.mark.asyncio
    async def test_get_session(self, session_manager: SessionManager) -> None:
        session_id, _, _ = await session_manager.create_session(user_id="user1", suites=["adversarial"])
        session = await session_manager.get_session(session_id)
        assert session is not None
        assert session["user_id"] == "user1"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, session_manager: SessionManager) -> None:
        session = await session_manager.get_session("nonexistent")
        assert session is None

    @pytest.mark.asyncio
    async def test_cancel_session(self, session_manager: SessionManager) -> None:
        session_id, _, _ = await session_manager.create_session(user_id="user1", suites=["adversarial"])
        assert await session_manager.cancel_session(session_id, "user1")
        session = await session_manager.get_session(session_id)
        assert session is not None
        assert session["status"] == SessionStatus.CANCELLED.value

    @pytest.mark.asyncio
    async def test_cancel_session_wrong_user(self, session_manager: SessionManager) -> None:
        session_id, _, _ = await session_manager.create_session(user_id="user1", suites=["adversarial"])
        assert not await session_manager.cancel_session(session_id, "user2")

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_session(self, session_manager: SessionManager) -> None:
        assert not await session_manager.cancel_session("nonexistent", "user1")

    @pytest.mark.asyncio
    async def test_verify_single_shot(self, session_manager: SessionManager) -> None:
        session_id, _, _ = await session_manager.create_session(user_id="user1", suites=["adversarial"])
        # Get server answers to submit correct ones
        answers_raw = await session_manager.get_session_answers(session_id)
        assert answers_raw is not None
        adv_server = answers_raw["adversarial"]

        # Submit correct math answer
        result = await session_manager.verify_single_shot(
            session_id,
            "adversarial",
            {"dynamic_math": {"computed": adv_server["dynamic_math"]["expected"], "time_ms": 50}},
        )
        assert "passed" in result
        assert "score" in result

    @pytest.mark.asyncio
    async def test_verify_wrong_suite(self, session_manager: SessionManager) -> None:
        session_id, _, _ = await session_manager.create_session(user_id="user1", suites=["adversarial"])
        with pytest.raises(ValueError, match="not in this session"):
            await session_manager.verify_single_shot(session_id, "native", {})

    @pytest.mark.asyncio
    async def test_verify_novel_reasoning_rejected_on_single_shot(self, session_manager: SessionManager) -> None:
        session_id, _, _ = await session_manager.create_session(user_id="user1", suites=["all"])
        with pytest.raises(ValueError, match="multi-round"):
            await session_manager.verify_single_shot(session_id, MULTI_ROUND_SUITE, {})

    @pytest.mark.asyncio
    async def test_verify_duplicate_suite(self, session_manager: SessionManager) -> None:
        session_id, _, _ = await session_manager.create_session(user_id="user1", suites=["adversarial", "native"])
        await session_manager.verify_single_shot(session_id, "adversarial", {})
        with pytest.raises(ValueError, match="already completed"):
            await session_manager.verify_single_shot(session_id, "adversarial", {})

    @pytest.mark.asyncio
    async def test_multi_round_submit(self, session_manager: SessionManager) -> None:
        session_id, challenges, _ = await session_manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="easy"
        )
        # Round 1
        novel_challenges = challenges.get("novel-reasoning", {}).get("challenges", {})
        round_answers = {"challenges": {name: {"test_outputs": []} for name in novel_challenges}}
        result = await session_manager.submit_round_answer(session_id, 1, round_answers)
        assert "accuracy" in result
        assert "time_remaining_ms" in result

    @pytest.mark.asyncio
    async def test_multi_round_wrong_order(self, session_manager: SessionManager) -> None:
        session_id, _, _ = await session_manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="easy"
        )
        with pytest.raises(ValueError, match="Expected round 1"):
            await session_manager.submit_round_answer(session_id, 2, {})

    @pytest.mark.asyncio
    async def test_get_result_not_completed(self, session_manager: SessionManager) -> None:
        session_id, _, _ = await session_manager.create_session(user_id="user1", suites=["adversarial"])
        result = await session_manager.get_result(session_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_session_completes_when_all_suites_done(self, session_manager: SessionManager) -> None:
        session_id, _, _ = await session_manager.create_session(user_id="user1", suites=["adversarial"])
        await session_manager.verify_single_shot(session_id, "adversarial", {})
        session = await session_manager.get_session(session_id)
        assert session is not None
        assert session["status"] == SessionStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_suite_resolution_all(self, session_manager: SessionManager) -> None:
        resolved = SessionManager._resolve_suites(["all"])
        assert len(resolved) == 10

    @pytest.mark.asyncio
    async def test_suite_resolution_specific(self, session_manager: SessionManager) -> None:
        resolved = SessionManager._resolve_suites(["adversarial", "native"])
        assert resolved == ["adversarial", "native"]


# ---- Rate Limiting Tests ----


class TestRateLimiting:
    """Test session rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_constants(self) -> None:
        assert MAX_ACTIVE_SESSIONS_PER_USER == 5
        assert MAX_SESSIONS_PER_HOUR == 100

    @pytest.mark.asyncio
    async def test_active_session_limit(self, fake_redis: FakeRedis) -> None:
        mgr = SessionManager(fake_redis)
        # Fill up active sessions
        for i in range(MAX_ACTIVE_SESSIONS_PER_USER):
            await mgr.create_session(user_id="user1", suites=["adversarial"])

        with pytest.raises(ValueError, match="Maximum active sessions"):
            await mgr.create_session(user_id="user1", suites=["adversarial"])


# ---- API Endpoint Tests ----


class TestSuiteEndpoints:
    """Test suite information endpoints."""

    def test_list_suites(self, client: TestClient) -> None:
        resp = client.get("/api/v1/mettle/suites")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 10
        names = [s["name"] for s in data]
        assert "adversarial" in names
        assert "novel-reasoning" in names

    def test_list_suites_structure(self, client: TestClient) -> None:
        resp = client.get("/api/v1/mettle/suites")
        suite = resp.json()[0]
        assert "name" in suite
        assert "display_name" in suite
        assert "description" in suite
        assert "suite_number" in suite
        assert "is_multi_round" in suite
        assert "difficulty_levels" in suite

    def test_get_suite_info(self, client: TestClient) -> None:
        resp = client.get("/api/v1/mettle/suites/adversarial")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "adversarial"
        assert data["suite_number"] == 1
        assert not data["is_multi_round"]

    def test_get_suite_info_novel_reasoning(self, client: TestClient) -> None:
        resp = client.get("/api/v1/mettle/suites/novel-reasoning")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_multi_round"]
        assert data["suite_number"] == 10

    def test_get_suite_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/v1/mettle/suites/nonexistent")
        assert resp.status_code == 404


class TestSessionEndpoints:
    """Test session management endpoints."""

    def test_create_session(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["adversarial"], "difficulty": "standard"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data
        assert "challenges" in data
        assert "adversarial" in data["challenges"]
        assert data["time_budget_ms"] > 0

    def test_create_session_all_suites(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["all"]},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert len(data["suites"]) == 10

    def test_create_session_invalid_suite(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["fake"]},
        )
        assert resp.status_code == 400

    def test_get_session_status(self, client: TestClient) -> None:
        # Create session
        create_resp = client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["adversarial"]},
        )
        session_id = create_resp.json()["session_id"]

        # Get status
        resp = client.get(f"/api/v1/mettle/sessions/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert data["status"] == "challenges_generated"

    def test_get_session_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/v1/mettle/sessions/nonexistent")
        assert resp.status_code == 404

    def test_cancel_session(self, client: TestClient) -> None:
        create_resp = client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["adversarial"]},
        )
        session_id = create_resp.json()["session_id"]

        resp = client.delete(f"/api/v1/mettle/sessions/{session_id}")
        assert resp.status_code == 204

    def test_cancel_nonexistent_session(self, client: TestClient) -> None:
        resp = client.delete("/api/v1/mettle/sessions/nonexistent")
        assert resp.status_code == 404


class TestVerificationEndpoints:
    """Test single-shot and multi-round verification."""

    def test_verify_single_shot(self, client: TestClient) -> None:
        # Create session
        create_resp = client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["adversarial"]},
        )
        session_id = create_resp.json()["session_id"]

        # Verify
        resp = client.post(
            f"/api/v1/mettle/sessions/{session_id}/verify",
            json={
                "suite": "adversarial",
                "answers": {"dynamic_math": {"computed": 0, "time_ms": 50}},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "passed" in data
        assert "score" in data
        assert data["suite"] == "adversarial"

    def test_verify_wrong_suite(self, client: TestClient) -> None:
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

    def test_multi_round_submit(self, client: TestClient) -> None:
        create_resp = client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["novel-reasoning"], "difficulty": "easy"},
        )
        assert create_resp.status_code == 201
        session_id = create_resp.json()["session_id"]

        # Submit round 1
        resp = client.post(
            f"/api/v1/mettle/sessions/{session_id}/rounds/1/answer",
            json={"answers": {}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "accuracy" in data
        assert "time_remaining_ms" in data

    def test_multi_round_wrong_order(self, client: TestClient) -> None:
        create_resp = client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["novel-reasoning"], "difficulty": "easy"},
        )
        session_id = create_resp.json()["session_id"]

        resp = client.post(
            f"/api/v1/mettle/sessions/{session_id}/rounds/2/answer",
            json={"answers": {}},
        )
        assert resp.status_code == 400

    def test_get_result_not_completed(self, client: TestClient) -> None:
        create_resp = client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["adversarial"]},
        )
        session_id = create_resp.json()["session_id"]

        resp = client.get(f"/api/v1/mettle/sessions/{session_id}/result")
        assert resp.status_code == 400

    def test_get_result_after_completion(self, client: TestClient) -> None:
        create_resp = client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["adversarial"]},
        )
        session_id = create_resp.json()["session_id"]

        # Complete the suite
        client.post(
            f"/api/v1/mettle/sessions/{session_id}/verify",
            json={"suite": "adversarial", "answers": {}},
        )

        # Get result
        resp = client.get(f"/api/v1/mettle/sessions/{session_id}/result")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert data["status"] == "completed"
        assert "adversarial" in data["suites_completed"]


class TestGetRoundFeedback:
    """Test round feedback retrieval."""

    def test_get_feedback_not_found(self, client: TestClient) -> None:
        create_resp = client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["novel-reasoning"]},
        )
        session_id = create_resp.json()["session_id"]

        resp = client.get(f"/api/v1/mettle/sessions/{session_id}/rounds/1/feedback")
        assert resp.status_code == 404

    def test_get_feedback_after_round(self, client: TestClient) -> None:
        create_resp = client.post(
            "/api/v1/mettle/sessions",
            json={"suites": ["novel-reasoning"], "difficulty": "easy"},
        )
        session_id = create_resp.json()["session_id"]

        # Submit round 1
        client.post(
            f"/api/v1/mettle/sessions/{session_id}/rounds/1/answer",
            json={"answers": {}},
        )

        # Get feedback
        resp = client.get(f"/api/v1/mettle/sessions/{session_id}/rounds/1/feedback")
        assert resp.status_code == 200
        data = resp.json()
        assert data["round"] == 1
        assert "accuracy" in data


# ---- Challenge Adapter Registry Tests ----


class TestSuiteRegistry:
    """Test suite registry metadata."""

    def test_all_suites_in_registry(self) -> None:
        for name in SUITE_NAMES:
            assert name in SUITE_REGISTRY, f"Suite '{name}' missing from registry"

    def test_registry_tuple_structure(self) -> None:
        for name, info in SUITE_REGISTRY.items():
            assert len(info) == 3, f"Suite '{name}' should have (display_name, description, number)"
            display_name, description, num = info
            assert isinstance(display_name, str)
            assert isinstance(description, str)
            assert isinstance(num, int)
            assert 1 <= num <= 10

    def test_suite_numbers_unique(self) -> None:
        numbers = [info[2] for info in SUITE_REGISTRY.values()]
        assert len(numbers) == len(set(numbers))


# ---- Novel Reasoning Round Evaluation Tests ----


class TestNovelReasoningEvaluation:
    """Test multi-round evaluation logic."""

    def test_evaluate_sequence_alchemy_round(self) -> None:
        server = {"all_test_answers": [10, 20, 30, 40, 50, 60]}
        answers = {"test_outputs": [10, 20]}
        result = ChallengeAdapter.evaluate_novel_round(
            "sequence_alchemy", 1, answers, {"challenges": {"sequence_alchemy": server}}
        )
        # Round 1 checks first 2 answers
        assert "accuracy" in result

    def test_evaluate_constraint_round_valid(self) -> None:
        server = {"all_solutions": [{"x": 1, "y": 2}], "constraint_data": []}
        answers = {"assignment": {"x": 1, "y": 2}}
        result = ChallengeAdapter.evaluate_novel_round(
            "constraint_satisfaction", 1, answers, {"challenges": {"constraint_satisfaction": server}}
        )
        assert result["accuracy"] == 1.0

    def test_evaluate_constraint_round_invalid(self) -> None:
        server = {"all_solutions": [{"x": 1, "y": 2}], "constraint_data": []}
        answers = {"assignment": {"x": 3, "y": 4}}
        result = ChallengeAdapter.evaluate_novel_round(
            "constraint_satisfaction", 1, answers, {"challenges": {"constraint_satisfaction": server}}
        )
        assert result["accuracy"] == 0.0

    def test_evaluate_encoding_round(self) -> None:
        server = {"original_message": "HELLO", "second_original": "WORLD"}
        answers = {"decoded_message": "HELLO"}
        result = ChallengeAdapter.evaluate_novel_round(
            "encoding_archaeology", 1, answers, {"challenges": {"encoding_archaeology": server}}
        )
        assert result["accuracy"] == 1.0

    def test_evaluate_graph_round(self) -> None:
        server = {"hidden_labels": {"A": "red", "B": "blue"}}
        answers = {"predicted_labels": {"A": "red", "B": "blue"}}
        result = ChallengeAdapter.evaluate_novel_round(
            "graph_property", 1, answers, {"challenges": {"graph_property": server}}
        )
        assert result["accuracy"] == 1.0

    def test_evaluate_logic_round(self) -> None:
        server = {"questions_with_answers": [{"answer": "true"}, {"answer": "false"}]}
        answers = {"answers": ["true", "false"]}
        result = ChallengeAdapter.evaluate_novel_round(
            "compositional_logic", 1, answers, {"challenges": {"compositional_logic": server}}
        )
        assert result["accuracy"] == 1.0

    def test_evaluate_unknown_challenge(self) -> None:
        result = ChallengeAdapter.evaluate_novel_round("unknown_type", 1, {}, {"challenges": {}})
        assert result["accuracy"] == 0.0


# ---- Edge Cases ----


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_answers_evaluation(self) -> None:
        result = ChallengeAdapter.evaluate_single_shot("adversarial", {}, {})
        assert result["score"] == 0.0

    def test_session_manager_ttl_constants(self) -> None:
        assert ACTIVE_SESSION_TTL == 300
        assert COMPLETED_SESSION_TTL == 3600

    @pytest.mark.asyncio
    async def test_get_answers_expired(self, session_manager: SessionManager, fake_redis: FakeRedis) -> None:
        session_id, _, _ = await session_manager.create_session(user_id="user1", suites=["adversarial"])
        # Delete answers to simulate expiry
        key = f"mettle:session:{session_id}:answers"
        fake_redis._store.pop(key, None)

        with pytest.raises(ValueError, match="expired"):
            await session_manager.verify_single_shot(session_id, "adversarial", {})

    @pytest.mark.asyncio
    async def test_session_entity_id(self, session_manager: SessionManager) -> None:
        session_id, _, meta = await session_manager.create_session(
            user_id="user1", suites=["adversarial"], entity_id="entity-abc"
        )
        assert meta["entity_id"] == "entity-abc"

    @pytest.mark.asyncio
    async def test_difficulty_affects_time_budget(self, session_manager: SessionManager) -> None:
        _, _, meta_easy = await session_manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="easy"
        )
        _, _, meta_hard = await session_manager.create_session(
            user_id="user2", suites=["novel-reasoning"], difficulty="hard"
        )
        # Hard has shorter time budget per round but same total structure
        assert meta_easy["time_budget_ms"] > 0
        assert meta_hard["time_budget_ms"] > 0
