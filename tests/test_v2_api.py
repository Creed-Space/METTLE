"""Comprehensive tests for METTLE API: session_manager.py and router.py.

Targets 100% coverage for:
- mettle/session_manager.py (~215 statements)
- mettle/router.py (~142 statements)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Mock scripts.engine BEFORE any session_manager imports to avoid numpy dep
# ---------------------------------------------------------------------------
_DIFFICULTY_PARAMS = {
    "easy": {"num_types": 2, "time_budget_s": 45, "num_rounds": 2},
    "standard": {"num_types": 3, "time_budget_s": 30, "num_rounds": 3},
    "hard": {"num_types": 3, "time_budget_s": 20, "num_rounds": 3},
}

_mock_engine = MagicMock()
_mock_engine.NovelReasoningChallenges = MagicMock()
_mock_engine.NovelReasoningChallenges.DIFFICULTY_PARAMS = _DIFFICULTY_PARAMS
_mock_engine.IterationCurveAnalyzer = MagicMock()
_mock_engine.IterationCurveAnalyzer.analyze_curve = MagicMock(
    return_value={
        "overall": 0.8,
        "signature": "AI",
        "time_trend": 0.7,
        "improvement": 0.6,
        "feedback_responsiveness": 0.5,
        "round1_suspicion": 0.0,
    }
)
sys.modules.setdefault("scripts.engine", _mock_engine)

from mettle.api_models import (  # noqa: E402
    MULTI_ROUND_SUITE,
    SUITE_NAMES,
    SessionStatus,
)
from mettle.auth import AuthenticatedUser, require_authenticated_user  # noqa: E402
from mettle.router import router  # noqa: E402
from mettle.session_manager import (  # noqa: E402
    COMPLETED_SESSION_TTL,
    MAX_ACTIVE_SESSIONS_PER_USER,
    MAX_SESSIONS_PER_HOUR,
    SessionManager,
    _key,
    _rate_key,
)


# ---------------------------------------------------------------------------
# FakeRedis / FakePipeline
# ---------------------------------------------------------------------------


class FakeRedis:
    """Minimal async Redis mock for testing."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self._sets: dict[str, set[str]] = {}

    async def get(self, key: str) -> bytes | None:
        return self._data.get(key)

    async def setex(self, key: str, ttl: int, value: Any) -> None:
        if isinstance(value, bytes):
            self._data[key] = value
        elif isinstance(value, str):
            self._data[key] = value.encode()
        else:
            self._data[key] = str(value).encode()

    async def scard(self, key: str) -> int:
        return len(self._sets.get(key, set()))

    async def sadd(self, key: str, *members: str) -> None:
        if key not in self._sets:
            self._sets[key] = set()
        self._sets[key].update(members)

    async def srem(self, key: str, *members: str) -> None:
        if key in self._sets:
            self._sets[key] -= set(members)

    async def incr(self, key: str) -> int:
        val = self._data.get(key, b"0")
        new_val = int(val) + 1
        self._data[key] = str(new_val).encode()
        return new_val

    async def expire(self, key: str, ttl: int) -> None:
        pass

    def pipeline(self) -> "FakePipeline":
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._ops: list[tuple[str, ...]] = []

    def setex(self, key: str, ttl: int, value: Any) -> "FakePipeline":
        self._ops.append(("setex", key, str(ttl), value))
        return self

    def sadd(self, key: str, *members: str) -> "FakePipeline":
        self._ops.append(("sadd", key, *members))
        return self

    def expire(self, key: str, ttl: int) -> "FakePipeline":
        self._ops.append(("expire", key, str(ttl)))
        return self

    def incr(self, key: str) -> "FakePipeline":
        self._ops.append(("incr", key))
        return self

    def srem(self, key: str, *members: str) -> "FakePipeline":
        self._ops.append(("srem", key, *members))
        return self

    async def execute(self) -> None:
        for op in self._ops:
            if op[0] == "setex":
                await self._redis.setex(op[1], int(op[2]), op[3])
            elif op[0] == "sadd":
                await self._redis.sadd(op[1], *op[2:])
            elif op[0] == "incr":
                await self._redis.incr(op[1])
            elif op[0] == "srem":
                await self._redis.srem(op[1], *op[2:])
        self._ops = []


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_USER = "test-user"
OTHER_USER = "other-user"

_CLIENT_STUB = {"suite": "stub", "challenges": {"q1": {}}}
_SERVER_STUB = {"q1": {"expected": 42}}
_NOVEL_SERVER_STUB = {
    "time_budget_s": 30,
    "num_rounds": 3,
    "pass_threshold": 0.65,
    "challenges": {"seq": {"all_test_answers": [1, 2]}},
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture()
def mgr(fake_redis: FakeRedis) -> SessionManager:
    return SessionManager(fake_redis)


@pytest.fixture(autouse=True)
def _ensure_mock_engine():
    """Ensure scripts.engine is mocked for this module's tests, restore after."""
    saved = sys.modules.get("scripts.engine")
    sys.modules["scripts.engine"] = _mock_engine
    yield
    if saved is not None:
        sys.modules["scripts.engine"] = saved
    else:
        sys.modules.pop("scripts.engine", None)


@pytest.fixture(autouse=True)
def _patch_challenge_adapter():
    """Patch ChallengeAdapter methods for ALL tests to avoid real challenge generation."""
    methods = {
        "generate_adversarial": (_CLIENT_STUB, _SERVER_STUB),
        "generate_native": (_CLIENT_STUB, _SERVER_STUB),
        "generate_self_reference": (_CLIENT_STUB, _SERVER_STUB),
        "generate_social": (_CLIENT_STUB, _SERVER_STUB),
        "generate_inverse_turing": (_CLIENT_STUB, _SERVER_STUB),
        "generate_anti_thrall": (_CLIENT_STUB, _SERVER_STUB),
        "generate_agency": (_CLIENT_STUB, _SERVER_STUB),
        "generate_counter_coaching": (_CLIENT_STUB, _SERVER_STUB),
        "generate_intent_provenance": (_CLIENT_STUB, _SERVER_STUB),
        "generate_novel_reasoning": (_CLIENT_STUB, _NOVEL_SERVER_STUB),
    }
    started = []
    for method, retval in methods.items():
        p = patch(f"mettle.session_manager.ChallengeAdapter.{method}", return_value=retval)
        p.start()
        started.append(p)
    p1 = patch(
        "mettle.session_manager.ChallengeAdapter.evaluate_single_shot",
        return_value={"passed": True, "score": 0.9, "details": {}},
    )
    p1.start()
    started.append(p1)
    p2 = patch(
        "mettle.session_manager.ChallengeAdapter.evaluate_novel_round",
        return_value={"accuracy": 0.8, "errors": []},
    )
    p2.start()
    started.append(p2)
    yield
    for p in started:
        p.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_session(
    mgr: SessionManager,
    suites: list[str] | None = None,
    user_id: str = TEST_USER,
    difficulty: str = "standard",
    vcp_token: str | None = None,
) -> tuple[str, dict, dict]:
    """Create a session. ChallengeAdapter is patched by autouse fixture."""
    return await mgr.create_session(
        user_id=user_id,
        suites=suites or ["adversarial"],
        difficulty=difficulty,
        vcp_token=vcp_token,
    )


def _store_session(fake_redis: FakeRedis, session_id: str, data: dict) -> None:
    """Synchronously store a session in FakeRedis."""
    asyncio.get_event_loop().run_until_complete(
        fake_redis.setex(_key(session_id), 300, json.dumps(data))
    )


def _make_foreign_session(
    session_id: str,
    user_id: str = OTHER_USER,
    status: str = SessionStatus.CHALLENGES_GENERATED.value,
    **extra: Any,
) -> dict:
    now = datetime.now(tz=timezone.utc)
    base = {
        "session_id": session_id,
        "user_id": user_id,
        "suites": ["adversarial"],
        "status": status,
        "created_at": now.isoformat(),
        "expires_at": now.isoformat(),
        "current_round": 0,
        "suites_completed": [],
        "start_time": None,
        "suite_results": {},
        "round_data": [],
    }
    base.update(extra)
    return base


# ===================================================================
# SessionManager unit tests
# ===================================================================


class TestKeyBuilders:
    def test_key_no_suffix(self):
        assert _key("abc") == "mettle:session:abc"

    def test_key_with_suffix(self):
        assert _key("abc", "answers") == "mettle:session:abc:answers"

    def test_rate_key(self):
        assert _rate_key("u1", "active") == "mettle:rate:u1:active"


class TestResolveSuites:
    def test_all_resolves_to_full_list(self):
        assert SessionManager._resolve_suites(["all"]) == list(SUITE_NAMES)

    def test_valid_list(self):
        assert SessionManager._resolve_suites(["adversarial", "native"]) == ["adversarial", "native"]

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown suites"):
            SessionManager._resolve_suites(["adversarial", "bogus"])


class TestCreateSession:
    @pytest.mark.asyncio
    async def test_create_single_suite(self, mgr, fake_redis):
        sid, challenges, meta = await _create_session(mgr, ["adversarial"])
        assert isinstance(sid, str) and len(sid) > 10
        assert "adversarial" in challenges
        assert meta["status"] == SessionStatus.CHALLENGES_GENERATED.value
        assert meta["suites"] == ["adversarial"]
        assert meta["time_budget_ms"] == 30000

    @pytest.mark.asyncio
    async def test_create_all_suites(self, mgr, fake_redis):
        sid, challenges, meta = await _create_session(mgr, ["all"])
        assert meta["suites"] == list(SUITE_NAMES)
        assert meta["time_budget_ms"] > 0

    @pytest.mark.asyncio
    async def test_create_with_novel_reasoning(self, mgr, fake_redis):
        sid, challenges, meta = await _create_session(mgr, ["adversarial", MULTI_ROUND_SUITE])
        assert MULTI_ROUND_SUITE in meta["suites"]
        assert meta["time_budget_ms"] == 30 * 1000 + 2 * 30000

    @pytest.mark.asyncio
    async def test_create_intent_provenance_with_vcp_token(self, mgr, fake_redis):
        sid, challenges, meta = await _create_session(
            mgr,
            ["intent-provenance"],
            vcp_token="VCP:3.1:agent-42\nC:test@1.0\nP:advisor:4",
        )
        assert "intent-provenance" in meta["suites"]

    @pytest.mark.asyncio
    async def test_create_invalid_suites(self, mgr):
        with pytest.raises(ValueError, match="Unknown suites"):
            await mgr.create_session(user_id=TEST_USER, suites=["bogus_suite"])

    @pytest.mark.asyncio
    async def test_rate_limit_active_sessions(self, mgr, fake_redis):
        fake_redis._sets[_rate_key(TEST_USER, "active")] = {f"s{i}" for i in range(MAX_ACTIVE_SESSIONS_PER_USER)}
        with pytest.raises(ValueError, match="Maximum active sessions"):
            await _create_session(mgr)

    @pytest.mark.asyncio
    async def test_rate_limit_hourly(self, mgr, fake_redis):
        fake_redis._data[_rate_key(TEST_USER, "hourly")] = str(MAX_SESSIONS_PER_HOUR).encode()
        with pytest.raises(ValueError, match="Hourly session limit"):
            await _create_session(mgr)

    @pytest.mark.asyncio
    async def test_unknown_suite_in_generators_raises(self, mgr, fake_redis):
        with patch.object(SessionManager, "_resolve_suites", return_value=["unknown-suite"]):
            with pytest.raises(ValueError, match="Unknown suite"):
                await mgr.create_session(user_id=TEST_USER, suites=["unknown-suite"])

    @pytest.mark.asyncio
    async def test_session_stored_in_redis(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr)
        raw = await fake_redis.get(_key(sid))
        assert raw is not None
        assert json.loads(raw)["session_id"] == sid

    @pytest.mark.asyncio
    async def test_answers_stored_in_redis(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr)
        assert await fake_redis.get(_key(sid, "answers")) is not None


class TestGetSession:
    @pytest.mark.asyncio
    async def test_get_existing(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr)
        session = await mgr.get_session(sid)
        assert session is not None and session["session_id"] == sid

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, mgr):
        assert await mgr.get_session("nonexistent") is None


class TestGetSessionAnswers:
    @pytest.mark.asyncio
    async def test_get_existing(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr)
        assert await mgr.get_session_answers(sid) is not None

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, mgr):
        assert await mgr.get_session_answers("nonexistent") is None


class TestCancelSession:
    @pytest.mark.asyncio
    async def test_cancel_success(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr)
        assert await mgr.cancel_session(sid, TEST_USER) is True
        session = await mgr.get_session(sid)
        assert session["status"] == SessionStatus.CANCELLED.value

    @pytest.mark.asyncio
    async def test_cancel_wrong_user(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr)
        assert await mgr.cancel_session(sid, OTHER_USER) is False

    @pytest.mark.asyncio
    async def test_cancel_not_found(self, mgr):
        assert await mgr.cancel_session("nonexistent", TEST_USER) is False

    @pytest.mark.asyncio
    async def test_cancel_already_completed(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr)
        session = await mgr.get_session(sid)
        session["status"] = SessionStatus.COMPLETED.value
        await fake_redis.setex(_key(sid), COMPLETED_SESSION_TTL, json.dumps(session))
        assert await mgr.cancel_session(sid, TEST_USER) is False

    @pytest.mark.asyncio
    async def test_cancel_already_cancelled(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr)
        session = await mgr.get_session(sid)
        session["status"] = SessionStatus.CANCELLED.value
        await fake_redis.setex(_key(sid), COMPLETED_SESSION_TTL, json.dumps(session))
        assert await mgr.cancel_session(sid, TEST_USER) is False


class TestVerifySingleShot:
    @pytest.mark.asyncio
    async def test_verify_success(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, ["adversarial"])
        result = await mgr.verify_single_shot(sid, "adversarial", {"q1": 42})
        assert result["passed"] is True
        assert result["score"] == 0.9

    @pytest.mark.asyncio
    async def test_verify_starts_timing(self, mgr, fake_redis):
        # Use two suites so the first verify puts us IN_PROGRESS (not COMPLETED)
        sid, _, _ = await _create_session(mgr, ["adversarial", "native"])
        await mgr.verify_single_shot(sid, "adversarial", {})
        session = await mgr.get_session(sid)
        assert session["start_time"] is not None
        assert session["status"] == SessionStatus.IN_PROGRESS.value

    @pytest.mark.asyncio
    async def test_verify_completes_all_suites(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, ["adversarial", "native"])
        await mgr.verify_single_shot(sid, "adversarial", {})
        await mgr.verify_single_shot(sid, "native", {})
        session = await mgr.get_session(sid)
        assert session["status"] == SessionStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_verify_session_not_found(self, mgr):
        with pytest.raises(ValueError, match="Session not found"):
            await mgr.verify_single_shot("nonexistent", "adversarial", {})

    @pytest.mark.asyncio
    async def test_verify_wrong_state(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, ["adversarial"])
        session = await mgr.get_session(sid)
        session["status"] = SessionStatus.COMPLETED.value
        await fake_redis.setex(_key(sid), 300, json.dumps(session))
        with pytest.raises(ValueError, match="not in verifiable state"):
            await mgr.verify_single_shot(sid, "adversarial", {})

    @pytest.mark.asyncio
    async def test_verify_suite_not_in_session(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, ["adversarial"])
        with pytest.raises(ValueError, match="not in this session"):
            await mgr.verify_single_shot(sid, "native", {})

    @pytest.mark.asyncio
    async def test_verify_suite_already_completed(self, mgr, fake_redis):
        # Use two suites so verifying the first doesn't complete the session
        sid, _, _ = await _create_session(mgr, ["adversarial", "native"])
        await mgr.verify_single_shot(sid, "adversarial", {})
        with pytest.raises(ValueError, match="already completed"):
            await mgr.verify_single_shot(sid, "adversarial", {})

    @pytest.mark.asyncio
    async def test_verify_novel_reasoning_rejected(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, [MULTI_ROUND_SUITE, "adversarial"])
        with pytest.raises(ValueError, match="multi-round endpoint"):
            await mgr.verify_single_shot(sid, MULTI_ROUND_SUITE, {})

    @pytest.mark.asyncio
    async def test_verify_answers_expired(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, ["adversarial"])
        del fake_redis._data[_key(sid, "answers")]
        with pytest.raises(ValueError, match="Session answers expired"):
            await mgr.verify_single_shot(sid, "adversarial", {})


class TestSubmitRoundAnswer:
    @pytest.mark.asyncio
    async def test_submit_round_success(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, [MULTI_ROUND_SUITE])
        result = await mgr.submit_round_answer(sid, 1, {"challenges": {"seq": {"test_outputs": [1]}}})
        assert result["round_num"] == 1
        assert result["accuracy"] == 0.8
        assert result["next_round_data"] is not None

    @pytest.mark.asyncio
    async def test_submit_round_starts_timing(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, [MULTI_ROUND_SUITE])
        await mgr.submit_round_answer(sid, 1, {"challenges": {"seq": {}}})
        session = await mgr.get_session(sid)
        assert session["start_time"] is not None
        assert session["status"] == SessionStatus.IN_PROGRESS.value

    @pytest.mark.asyncio
    async def test_submit_final_round_completes(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, [MULTI_ROUND_SUITE])
        await mgr.submit_round_answer(sid, 1, {"challenges": {"seq": {}}})
        await mgr.submit_round_answer(sid, 2, {"challenges": {"seq": {}}})
        result = await mgr.submit_round_answer(sid, 3, {"challenges": {"seq": {}}})
        assert result["next_round_data"] is None
        session = await mgr.get_session(sid)
        assert MULTI_ROUND_SUITE in session["suites_completed"]
        assert session["status"] == SessionStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_submit_round_not_found(self, mgr):
        with pytest.raises(ValueError, match="Session not found"):
            await mgr.submit_round_answer("nonexistent", 1, {})

    @pytest.mark.asyncio
    async def test_submit_round_no_novel_suite(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, ["adversarial"])
        with pytest.raises(ValueError, match="does not include novel-reasoning"):
            await mgr.submit_round_answer(sid, 1, {})

    @pytest.mark.asyncio
    async def test_submit_round_wrong_state(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, [MULTI_ROUND_SUITE])
        session = await mgr.get_session(sid)
        session["status"] = SessionStatus.COMPLETED.value
        await fake_redis.setex(_key(sid), 300, json.dumps(session))
        with pytest.raises(ValueError, match="not in answerable state"):
            await mgr.submit_round_answer(sid, 1, {})

    @pytest.mark.asyncio
    async def test_submit_round_wrong_sequence(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, [MULTI_ROUND_SUITE])
        with pytest.raises(ValueError, match="Expected round 1, got 2"):
            await mgr.submit_round_answer(sid, 2, {})

    @pytest.mark.asyncio
    async def test_submit_round_answers_expired(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, [MULTI_ROUND_SUITE])
        del fake_redis._data[_key(sid, "answers")]
        with pytest.raises(ValueError, match="Session answers expired"):
            await mgr.submit_round_answer(sid, 1, {})

    @pytest.mark.asyncio
    async def test_submit_round_uses_flat_answers(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, [MULTI_ROUND_SUITE])
        result = await mgr.submit_round_answer(sid, 1, {"seq": {"test_outputs": [1]}})
        assert result["round_num"] == 1


class TestGetRoundFeedback:
    @pytest.mark.asyncio
    async def test_get_feedback_found(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, [MULTI_ROUND_SUITE])
        await mgr.submit_round_answer(sid, 1, {"challenges": {"seq": {}}})
        feedback = await mgr.get_round_feedback(sid, 1)
        assert feedback is not None and feedback["round"] == 1

    @pytest.mark.asyncio
    async def test_get_feedback_not_found_round(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, [MULTI_ROUND_SUITE])
        assert await mgr.get_round_feedback(sid, 99) is None

    @pytest.mark.asyncio
    async def test_get_feedback_session_not_found(self, mgr):
        assert await mgr.get_round_feedback("nonexistent", 1) is None


class TestGetResult:
    @pytest.mark.asyncio
    async def test_get_result_completed(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, ["adversarial"])
        await mgr.verify_single_shot(sid, "adversarial", {})
        result = await mgr.get_result(sid)
        assert result is not None
        assert result["overall_passed"] is True
        assert result["session_id"] == sid

    @pytest.mark.asyncio
    async def test_get_result_completed_with_novel_reasoning(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, [MULTI_ROUND_SUITE])
        await mgr.submit_round_answer(sid, 1, {"challenges": {"seq": {}}})
        await mgr.submit_round_answer(sid, 2, {"challenges": {"seq": {}}})
        await mgr.submit_round_answer(sid, 3, {"challenges": {"seq": {}}})
        result = await mgr.get_result(sid)
        assert result is not None
        assert result["iteration_curve"] is not None

    @pytest.mark.asyncio
    async def test_get_result_not_completed(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, ["adversarial"])
        assert await mgr.get_result(sid) is None

    @pytest.mark.asyncio
    async def test_get_result_not_found(self, mgr):
        assert await mgr.get_result("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_result_no_start_time(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, ["adversarial"])
        session = await mgr.get_session(sid)
        session["status"] = SessionStatus.COMPLETED.value
        session["suite_results"] = {"adversarial": {"passed": True}}
        session["suites_completed"] = ["adversarial"]
        session["start_time"] = None
        await fake_redis.setex(_key(sid), COMPLETED_SESSION_TTL, json.dumps(session))
        result = await mgr.get_result(sid)
        assert result is not None and result["elapsed_ms"] == 0

    @pytest.mark.asyncio
    async def test_get_result_empty_results(self, mgr, fake_redis):
        sid, _, _ = await _create_session(mgr, ["adversarial"])
        session = await mgr.get_session(sid)
        session["status"] = SessionStatus.COMPLETED.value
        session["suite_results"] = {}
        session["suites_completed"] = []
        session["start_time"] = time.time()
        await fake_redis.setex(_key(sid), COMPLETED_SESSION_TTL, json.dumps(session))
        result = await mgr.get_result(sid)
        assert result["overall_passed"] is False


class TestAnalyzeIterationCurve:
    def test_analyze_curve(self, mgr):
        rd = [
            {"round": 1, "response_time_ms": 500, "accuracy": 0.4},
            {"round": 2, "response_time_ms": 400, "accuracy": 0.6},
            {"round": 3, "response_time_ms": 300, "accuracy": 0.8},
        ]
        result = mgr._analyze_iteration_curve(rd, {"pass_threshold": 0.65})
        assert result["passed"] is True
        assert result["score"] == 0.8

    def test_analyze_curve_script_signature(self, mgr):
        _mock_engine.IterationCurveAnalyzer.analyze_curve.return_value = {
            "overall": 0.9,
            "signature": "SCRIPT",
        }
        rd = [
            {"round": 1, "response_time_ms": 100, "accuracy": 0.99},
            {"round": 2, "response_time_ms": 100, "accuracy": 0.99},
        ]
        result = mgr._analyze_iteration_curve(rd, {"pass_threshold": 0.65})
        assert result["passed"] is False
        # Restore default
        _mock_engine.IterationCurveAnalyzer.analyze_curve.return_value = {
            "overall": 0.8,
            "signature": "AI",
        }

    def test_analyze_curve_below_threshold(self, mgr):
        _mock_engine.IterationCurveAnalyzer.analyze_curve.return_value = {
            "overall": 0.4,
            "signature": "AI",
        }
        rd = [
            {"round": 1, "response_time_ms": 500, "accuracy": 0.2},
            {"round": 2, "response_time_ms": 400, "accuracy": 0.3},
        ]
        result = mgr._analyze_iteration_curve(rd, {"pass_threshold": 0.65})
        assert result["passed"] is False
        _mock_engine.IterationCurveAnalyzer.analyze_curve.return_value = {
            "overall": 0.8,
            "signature": "AI",
        }


class TestCheckRateLimits:
    @pytest.mark.asyncio
    async def test_no_limits_hit(self, mgr, fake_redis):
        await mgr._check_rate_limits(TEST_USER)

    @pytest.mark.asyncio
    async def test_active_sessions_limit(self, mgr, fake_redis):
        fake_redis._sets[_rate_key(TEST_USER, "active")] = {f"s{i}" for i in range(MAX_ACTIVE_SESSIONS_PER_USER)}
        with pytest.raises(ValueError, match="Maximum active sessions"):
            await mgr._check_rate_limits(TEST_USER)

    @pytest.mark.asyncio
    async def test_hourly_limit(self, mgr, fake_redis):
        fake_redis._data[_rate_key(TEST_USER, "hourly")] = str(MAX_SESSIONS_PER_HOUR).encode()
        with pytest.raises(ValueError, match="Hourly session limit"):
            await mgr._check_rate_limits(TEST_USER)

    @pytest.mark.asyncio
    async def test_scard_returns_none(self, mgr, fake_redis):
        await mgr._check_rate_limits(TEST_USER)


# ===================================================================
# Router tests via TestClient
# ===================================================================


def _make_test_app(fake_redis: FakeRedis) -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.state.redis = fake_redis

    async def mock_auth():
        return AuthenticatedUser(user_id=TEST_USER)

    test_app.dependency_overrides[require_authenticated_user] = mock_auth
    return test_app


@pytest.fixture()
def test_app(fake_redis: FakeRedis) -> FastAPI:
    app = _make_test_app(fake_redis)
    yield app
    app.dependency_overrides.clear()


@pytest.fixture()
def client(test_app: FastAPI) -> TestClient:
    return TestClient(test_app)


def _create_via_api(client: TestClient, suites: list[str] | None = None) -> str:
    """Create a session via the API and return the session_id."""
    resp = client.post(
        "/api/mettle/sessions",
        json={"suites": suites or ["adversarial"], "difficulty": "standard"},
    )
    assert resp.status_code == 201, f"Session creation failed: {resp.json()}"
    return resp.json()["session_id"]


class TestRouterGetSessionManager:
    def test_redis_unavailable_returns_503(self):
        app = FastAPI()
        app.include_router(router)
        app.state.redis = None

        async def mock_auth():
            return AuthenticatedUser(user_id=TEST_USER)

        app.dependency_overrides[require_authenticated_user] = mock_auth
        c = TestClient(app)
        # Use an endpoint that actually depends on get_session_manager
        resp = c.post("/api/mettle/sessions", json={"suites": ["adversarial"]})
        assert resp.status_code == 503
        app.dependency_overrides.clear()


class TestRouterListSuites:
    def test_list_suites(self, client):
        resp = client.get("/api/mettle/suites")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 10
        names = [s["name"] for s in data]
        assert "adversarial" in names
        assert MULTI_ROUND_SUITE in names
        novel = [s for s in data if s["name"] == MULTI_ROUND_SUITE][0]
        assert novel["is_multi_round"] is True


class TestRouterGetSuiteInfo:
    def test_get_valid_suite(self, client):
        resp = client.get("/api/mettle/suites/adversarial")
        assert resp.status_code == 200
        assert resp.json()["name"] == "adversarial"
        assert resp.json()["suite_number"] == 1

    def test_get_invalid_suite(self, client):
        assert client.get("/api/mettle/suites/bogus").status_code == 404


class TestRouterCreateSession:
    def test_create_session_success(self, client):
        sid = _create_via_api(client, ["adversarial"])
        assert isinstance(sid, str) and len(sid) > 10

    def test_create_session_invalid_suites(self, client):
        resp = client.post("/api/mettle/sessions", json={"suites": ["bogus"]})
        assert resp.status_code == 400

    def test_create_session_unexpected_error(self, client):
        with patch.object(SessionManager, "create_session", side_effect=RuntimeError("boom")):
            resp = client.post("/api/mettle/sessions", json={"suites": ["adversarial"]})
            assert resp.status_code == 500


class TestRouterGetSessionStatus:
    def test_get_session_status_success(self, client):
        sid = _create_via_api(client)
        resp = client.get(f"/api/mettle/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == sid
        assert resp.json()["elapsed_ms"] == 0

    def test_get_session_not_found(self, client):
        assert client.get("/api/mettle/sessions/nonexistent").status_code == 404

    def test_get_session_wrong_user(self, client, fake_redis):
        data = _make_foreign_session("foreign-session")
        _store_session(fake_redis, "foreign-session", data)
        assert client.get("/api/mettle/sessions/foreign-session").status_code == 403

    def test_get_session_with_start_time(self, client, fake_redis):
        data = _make_foreign_session(
            "timed-session",
            user_id=TEST_USER,
            status=SessionStatus.IN_PROGRESS.value,
            start_time=time.time() - 5,
        )
        _store_session(fake_redis, "timed-session", data)
        resp = client.get("/api/mettle/sessions/timed-session")
        assert resp.status_code == 200
        assert resp.json()["elapsed_ms"] > 0


class TestRouterCancelSession:
    def test_cancel_success(self, client):
        sid = _create_via_api(client)
        assert client.delete(f"/api/mettle/sessions/{sid}").status_code == 204

    def test_cancel_not_found(self, client):
        assert client.delete("/api/mettle/sessions/nonexistent").status_code == 404


class TestRouterVerifySingleShot:
    def test_verify_success(self, client):
        sid = _create_via_api(client, ["adversarial"])
        resp = client.post(
            f"/api/mettle/sessions/{sid}/verify",
            json={"suite": "adversarial", "answers": {"q1": 42}},
        )
        assert resp.status_code == 200
        assert resp.json()["passed"] is True
        assert resp.json()["suite"] == "adversarial"

    def test_verify_session_not_found(self, client):
        resp = client.post(
            "/api/mettle/sessions/nonexistent/verify",
            json={"suite": "adversarial", "answers": {}},
        )
        assert resp.status_code == 404

    def test_verify_wrong_user(self, client, fake_redis):
        data = _make_foreign_session("foreign-verify")
        _store_session(fake_redis, "foreign-verify", data)
        resp = client.post(
            "/api/mettle/sessions/foreign-verify/verify",
            json={"suite": "adversarial", "answers": {}},
        )
        assert resp.status_code == 403

    def test_verify_value_error(self, client):
        sid = _create_via_api(client, ["adversarial"])
        resp = client.post(
            f"/api/mettle/sessions/{sid}/verify",
            json={"suite": "native", "answers": {}},
        )
        assert resp.status_code == 400

    def test_verify_unexpected_error(self, client):
        sid = _create_via_api(client, ["adversarial"])
        with patch.object(SessionManager, "verify_single_shot", side_effect=RuntimeError("boom")):
            resp = client.post(
                f"/api/mettle/sessions/{sid}/verify",
                json={"suite": "adversarial", "answers": {}},
            )
            assert resp.status_code == 500


class TestRouterSubmitRoundAnswer:
    def test_submit_round_success(self, client):
        sid = _create_via_api(client, [MULTI_ROUND_SUITE])
        resp = client.post(
            f"/api/mettle/sessions/{sid}/rounds/1/answer",
            json={"answers": {"challenges": {"seq": {}}}},
        )
        assert resp.status_code == 200
        assert resp.json()["round_num"] == 1

    def test_submit_round_not_found(self, client):
        resp = client.post(
            "/api/mettle/sessions/nonexistent/rounds/1/answer",
            json={"answers": {}},
        )
        assert resp.status_code == 404

    def test_submit_round_wrong_user(self, client, fake_redis):
        data = _make_foreign_session("foreign-round", suites=[MULTI_ROUND_SUITE])
        _store_session(fake_redis, "foreign-round", data)
        resp = client.post(
            "/api/mettle/sessions/foreign-round/rounds/1/answer",
            json={"answers": {}},
        )
        assert resp.status_code == 403

    def test_submit_round_value_error(self, client):
        sid = _create_via_api(client, [MULTI_ROUND_SUITE])
        resp = client.post(
            f"/api/mettle/sessions/{sid}/rounds/2/answer",
            json={"answers": {}},
        )
        assert resp.status_code == 400

    def test_submit_round_unexpected_error(self, client):
        sid = _create_via_api(client, [MULTI_ROUND_SUITE])
        with patch.object(SessionManager, "submit_round_answer", side_effect=RuntimeError("boom")):
            resp = client.post(
                f"/api/mettle/sessions/{sid}/rounds/1/answer",
                json={"answers": {}},
            )
            assert resp.status_code == 500


class TestRouterGetRoundFeedback:
    def test_get_feedback_success(self, client):
        sid = _create_via_api(client, [MULTI_ROUND_SUITE])
        client.post(
            f"/api/mettle/sessions/{sid}/rounds/1/answer",
            json={"answers": {"challenges": {"seq": {}}}},
        )
        resp = client.get(f"/api/mettle/sessions/{sid}/rounds/1/feedback")
        assert resp.status_code == 200

    def test_get_feedback_not_found_session(self, client):
        assert client.get("/api/mettle/sessions/nonexistent/rounds/1/feedback").status_code == 404

    def test_get_feedback_wrong_user(self, client, fake_redis):
        data = _make_foreign_session("foreign-feedback", status=SessionStatus.IN_PROGRESS.value)
        data["round_data"] = [{"round": 1, "accuracy": 0.5}]
        _store_session(fake_redis, "foreign-feedback", data)
        assert client.get("/api/mettle/sessions/foreign-feedback/rounds/1/feedback").status_code == 403

    def test_get_feedback_round_not_done(self, client):
        sid = _create_via_api(client, [MULTI_ROUND_SUITE])
        assert client.get(f"/api/mettle/sessions/{sid}/rounds/1/feedback").status_code == 404


class TestRouterGetSessionResult:
    def test_get_result_success(self, client):
        sid = _create_via_api(client, ["adversarial"])
        client.post(
            f"/api/mettle/sessions/{sid}/verify",
            json={"suite": "adversarial", "answers": {"q1": 42}},
        )
        resp = client.get(f"/api/mettle/sessions/{sid}/result")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == sid
        assert "overall_passed" in data
        assert "tier" in data

    def test_get_result_not_found(self, client):
        assert client.get("/api/mettle/sessions/nonexistent/result").status_code == 404

    def test_get_result_wrong_user(self, client, fake_redis):
        data = _make_foreign_session("foreign-result", status=SessionStatus.COMPLETED.value)
        _store_session(fake_redis, "foreign-result", data)
        assert client.get("/api/mettle/sessions/foreign-result/result").status_code == 403

    def test_get_result_not_completed(self, client):
        sid = _create_via_api(client, ["adversarial"])
        resp = client.get(f"/api/mettle/sessions/{sid}/result")
        assert resp.status_code == 400

    def test_get_result_with_vcp(self, client):
        sid = _create_via_api(client, ["adversarial"])
        client.post(
            f"/api/mettle/sessions/{sid}/verify",
            json={"suite": "adversarial", "answers": {"q1": 42}},
        )
        resp = client.get(f"/api/mettle/sessions/{sid}/result?include_vcp=true")
        assert resp.status_code == 200
        assert "vcp_attestation" in resp.json()

    def test_get_result_with_vcp_and_signing(self, client):
        sid = _create_via_api(client, ["adversarial"])
        client.post(
            f"/api/mettle/sessions/{sid}/verify",
            json={"suite": "adversarial", "answers": {"q1": 42}},
        )
        mock_signing = MagicMock()
        mock_signing.is_available.return_value = True
        mock_signing.sign_attestation = MagicMock(return_value="fake-sig")
        with patch.dict("sys.modules", {"mettle.signing": mock_signing}):
            resp = client.get(f"/api/mettle/sessions/{sid}/result?include_vcp=true")
        assert resp.status_code == 200


class TestRouterVCPKeys:
    def test_vcp_keys_no_signing(self, client):
        with patch.dict("sys.modules", {"mettle.signing": None}):
            resp = client.get("/api/mettle/.well-known/vcp-keys")
            assert resp.status_code == 200

    def test_vcp_keys_with_signing(self, client):
        mock_signing = MagicMock()
        mock_signing.get_public_key_info.return_value = {
            "key_id": "mettle-vcp-v1",
            "algorithm": "Ed25519",
            "public_key_pem": "fake-pem",
            "available": True,
        }
        with patch.dict("sys.modules", {"mettle.signing": mock_signing}):
            resp = client.get("/api/mettle/.well-known/vcp-keys")
            assert resp.status_code == 200
