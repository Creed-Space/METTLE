"""Comprehensive coverage tests for mettle/session_manager.py.

Covers gaps not addressed by test_mettle_api.py:
- Hourly rate limiting
- Cancel on completed/cancelled sessions
- verify_single_shot on expired/nonexistent/completed sessions
- submit_round_answer edge cases (no novel-reasoning, expired answers, completed session, final round)
- get_round_feedback for non-existent and existing rounds
- get_result with start_time, empty results, all passed, novel-reasoning results
- _analyze_iteration_curve with known data
- _key and _rate_key helpers
- Multiple suites with partial completion
- Time budget calculation with and without novel reasoning
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from redis.exceptions import RedisError

import mettle.session_manager as session_module
from mettle.api_models import (
    MULTI_ROUND_SUITE,
    SessionStatus,
)
from mettle.session_manager import (
    ACTIVE_SESSION_TTL,
    MAX_SESSIONS_PER_HOUR,
    SessionManager,
    SessionLockLostError,
    SessionRateLimitError,
    SessionStateError,
    _key,
    _presentation_key,
    _presentation_rate_key,
    _rate_key,
)

# ---- Fake Redis (copied from test_mettle_api.py) ----


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

    async def delete(self, key: str) -> int:
        return 1 if self._store.pop(key, None) is not None else 0

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

    async def smembers(self, key: str) -> set[str]:
        return set(self._sets.get(key, set()))

    async def incr(self, key: str) -> int:
        val = int(self._store.get(key, "0"))
        val += 1
        self._store[key] = str(val)
        return val

    async def decr(self, key: str) -> int:
        val = int(self._store.get(key, "0")) - 1
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

    def delete(self, key: str) -> FakeRedisPipeline:
        self._commands.append(("delete", (key,)))
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


class SemanticRedis(FakeRedis):
    """Redis double that executes the session Lua contracts atomically."""

    def __init__(self) -> None:
        super().__init__()
        self._mutex = asyncio.Lock()
        self._lock_expiry: dict[str, float] = {}
        self._zsets: dict[str, dict[str, int]] = {}
        self.fail_terminal = False
        self.corrupt_active_index = False
        self.renewals = 0

    def _purge_lock(self, key: str) -> None:
        expires = self._lock_expiry.get(key)
        if expires is not None and time.monotonic() >= expires:
            self._store.pop(key, None)
            self._lock_expiry.pop(key, None)

    async def get(self, key: str) -> str | None:
        self._purge_lock(key)
        return await super().get(key)

    async def set(
        self, key: str, value: str, *, nx: bool = False, ex: float | int | None = None
    ) -> bool:
        async with self._mutex:
            self._purge_lock(key)
            if nx and key in self._store:
                return False
            self._store[key] = value
            if ex is not None:
                self._lock_expiry[key] = time.monotonic() + float(ex)
            return True

    async def eval(self, script: str, numkeys: int, *args: Any) -> Any:
        keys = list(args[:numkeys])
        argv = list(args[numkeys:])
        async with self._mutex:
            if script == session_module._LOCK_CHECK_SCRIPT:
                self._purge_lock(keys[0])
                return int(self._store.get(keys[0]) == argv[0])
            if script == session_module._LOCK_RENEW_SCRIPT:
                self._purge_lock(keys[0])
                if self._store.get(keys[0]) != argv[0]:
                    return 0
                self._lock_expiry[keys[0]] = time.monotonic() + float(argv[1])
                self.renewals += 1
                return 1
            if script == session_module._LOCK_RELEASE_SCRIPT:
                self._purge_lock(keys[0])
                if self._store.get(keys[0]) != argv[0]:
                    return 0
                return await self.delete(keys[0])
            if script == session_module._ACTIVE_PERSIST_SCRIPT:
                self._purge_lock(keys[1])
                if self._store.get(keys[1]) != argv[2]:
                    return -1
                await self.setex(keys[0], int(argv[1]), str(argv[0]))
                return 1
            if script == session_module._TERMINAL_PERSIST_SCRIPT:
                self._purge_lock(keys[3])
                if self._store.get(keys[3]) != argv[3]:
                    return -1
                if self.corrupt_active_index:
                    return -2
                if self.fail_terminal:
                    raise RedisError("injected atomic terminal failure")
                await self.setex(keys[0], int(argv[1]), str(argv[0]))
                await self.srem(keys[1], str(argv[2]))
                await self.delete(keys[2])
                await self.delete(keys[4])
                return 1
            if script == session_module._PRESENTATION_CONSUME_SCRIPT:
                if self._store.get(keys[0]) != argv[0]:
                    return 0
                return await self.delete(keys[0])
            raise AssertionError("unexpected Lua script")

    def replace_lock_owner(self, session_id: str) -> None:
        key = _key(session_id, "lock")
        self._store[key] = "another-worker"
        self._lock_expiry[key] = time.monotonic() + 60


# ---- Fixtures ----


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def manager(fake_redis: FakeRedis) -> SessionManager:
    return SessionManager(fake_redis)


# ---- Helper to manually inject a session into Redis ----


async def _inject_session(
    redis: FakeRedis,
    session_id: str,
    session_data: dict[str, Any],
    answers: dict[str, Any] | None = None,
    ttl: int = ACTIVE_SESSION_TTL,
) -> None:
    """Directly inject a session into fake Redis for controlled test scenarios."""
    await redis.setex(_key(session_id), ttl, json.dumps(session_data))
    if answers is not None:
        await redis.setex(_key(session_id, "answers"), ttl, json.dumps(answers))


# ---- 17. _key helper ----


class TestKeyHelper:
    """Test the _key Redis key builder."""

    def test_key_without_suffix(self) -> None:
        result = _key("abc123")
        assert result == "mettle:session:abc123"

    def test_key_with_suffix(self) -> None:
        result = _key("abc123", "answers")
        assert result == "mettle:session:abc123:answers"

    def test_key_with_empty_suffix(self) -> None:
        result = _key("abc123", "")
        assert result == "mettle:session:abc123"

    def test_key_with_complex_session_id(self) -> None:
        sid = "xYz_-09Ab"
        result = _key(sid, "meta")
        assert result == f"mettle:session:{sid}:meta"


# ---- 18. _rate_key helper ----


class TestRateKeyHelper:
    """Test the _rate_key Redis key builder."""

    def test_rate_key_active(self) -> None:
        result = _rate_key("user1", "active")
        assert result == "mettle:rate:user1:active"

    def test_rate_key_hourly(self) -> None:
        result = _rate_key("user1", "hourly")
        assert result == "mettle:rate:user1:hourly"

    def test_rate_key_different_users(self) -> None:
        r1 = _rate_key("alice", "active")
        r2 = _rate_key("bob", "active")
        assert r1 != r2
        assert "alice" in r1
        assert "bob" in r2

    def test_configured_namespace_is_applied_to_all_keys(self, monkeypatch) -> None:
        from mettle.app_config import settings

        monkeypatch.setattr(settings, "redis_namespace", "mettle-staging")

        assert _key("abc123") == "mettle-staging:session:abc123"
        assert _rate_key("agent", "hourly") == "mettle-staging:rate:agent:hourly"


# ---- 1. Hourly rate limit ----


class TestHourlyRateLimit:
    """Test that hourly limit is enforced at MAX_SESSIONS_PER_HOUR."""

    @pytest.mark.asyncio
    async def test_hourly_rate_limit_blocks_at_max(self, fake_redis: FakeRedis) -> None:
        mgr = SessionManager(fake_redis)
        # Simulate hourly counter already at the limit by directly setting the key
        hourly_key = _rate_key("user1", "hourly")
        fake_redis._store[hourly_key] = str(MAX_SESSIONS_PER_HOUR)

        with pytest.raises(ValueError, match="Hourly session limit"):
            await mgr.create_session(user_id="user1", suites=["adversarial"])

    @pytest.mark.asyncio
    async def test_hourly_rate_limit_allows_below_max(
        self, fake_redis: FakeRedis
    ) -> None:
        mgr = SessionManager(fake_redis)
        # Set hourly counter just below the limit
        hourly_key = _rate_key("user1", "hourly")
        fake_redis._store[hourly_key] = str(MAX_SESSIONS_PER_HOUR - 1)

        # Should not raise
        session_id, _, _ = await mgr.create_session(
            user_id="user1", suites=["adversarial"]
        )
        assert session_id

    @pytest.mark.asyncio
    async def test_hourly_rate_limit_message_includes_count(
        self, fake_redis: FakeRedis
    ) -> None:
        mgr = SessionManager(fake_redis)
        hourly_key = _rate_key("user1", "hourly")
        fake_redis._store[hourly_key] = str(MAX_SESSIONS_PER_HOUR)

        with pytest.raises(ValueError, match=str(MAX_SESSIONS_PER_HOUR)):
            await mgr.create_session(user_id="user1", suites=["adversarial"])


class TestSessionSecurityBoundaries:
    @pytest.mark.asyncio
    async def test_duplicate_suites_are_rejected(self, manager: SessionManager) -> None:
        with pytest.raises(ValueError, match="Duplicate suites"):
            await manager.create_session(
                user_id="user1", suites=["adversarial", "adversarial"]
            )

    @pytest.mark.asyncio
    async def test_server_issued_time_budget_expires_single_shot(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        session_id, _, session = await manager.create_session(
            user_id="user1", suites=["adversarial"]
        )
        session["start_time"] = time.time() - 60
        session["time_budget_ms"] = 1000
        await fake_redis.setex(
            _key(session_id), ACTIVE_SESSION_TTL, json.dumps(session)
        )

        with pytest.raises(ValueError, match="time budget exceeded"):
            await manager.verify_single_shot(session_id, "adversarial", {})

        stored = await manager.get_session(session_id)
        assert stored is not None
        assert stored["status"] == SessionStatus.EXPIRED.value

    def test_active_ttl_cannot_extend_absolute_expiry(self) -> None:
        session = {
            "expires_at": datetime.fromtimestamp(
                time.time() + 10, tz=timezone.utc
            ).isoformat()
        }

        ttl = SessionManager._remaining_active_ttl(session)

        assert 1 <= ttl <= 10

    def test_active_ttl_preserves_budget_beyond_legacy_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(session_module.time, "time", lambda: 1_000.25)
        session = {
            "expires_at": datetime.fromtimestamp(1_400.0, tz=timezone.utc).isoformat()
        }

        assert SessionManager._remaining_active_ttl(session) == 400

    def test_human_iteration_signature_cannot_pass(self, monkeypatch) -> None:
        from scripts.engine import IterationCurveAnalyzer

        monkeypatch.setattr(
            IterationCurveAnalyzer,
            "analyze_curve",
            lambda _rounds: {"overall": 1.0, "signature": "HUMAN"},
        )

        result = SessionManager._analyze_iteration_curve(
            SessionManager.__new__(SessionManager),
            [{"round": 1, "response_time_ms": 1, "accuracy": 1.0}],
            {"pass_threshold": 0.65},
        )

        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_atomic_rate_reservation_rejects_active_limit(self) -> None:
        redis = AsyncMock()
        redis.eval.return_value = -1
        manager = SessionManager(redis)

        with pytest.raises(ValueError, match="Maximum active sessions"):
            await manager._reserve_rate_limits("user-1", "session-1")

        redis.eval.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_atomic_rate_reservation_rejects_hourly_limit(self) -> None:
        redis = AsyncMock()
        redis.eval.return_value = -2
        manager = SessionManager(redis)

        with pytest.raises(ValueError, match="Hourly session limit"):
            await manager._reserve_rate_limits("user-1", "session-1")

    @pytest.mark.asyncio
    async def test_session_transition_lock_rejects_concurrent_operation(self) -> None:
        redis = AsyncMock()
        redis.set.return_value = False
        manager = SessionManager(redis)

        with pytest.raises(ValueError, match="already in progress"):
            async with manager._session_lock("session-1"):
                pytest.fail("lock body must not execute")

    @pytest.mark.asyncio
    async def test_session_transition_lock_releases_only_by_token(self) -> None:
        redis = AsyncMock()
        redis.set.return_value = True
        manager = SessionManager(redis)

        async with manager._session_lock("session-1"):
            pass

        redis.set.assert_awaited_once()
        assert redis.eval.await_count == 2
        scripts = [call.args[0] for call in redis.eval.await_args_list]
        assert "redis.call('GET'" in scripts[0]
        assert "redis.call('DEL'" in scripts[1]


# ---- 2. Cancel already completed session ----


class TestCancelCompletedSession:
    """Test that cancelling a completed session returns False."""

    @pytest.mark.asyncio
    async def test_cancel_completed_session_returns_false(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        session_id, _, _ = await manager.create_session(
            user_id="user1", suites=["adversarial"]
        )
        # Complete the session by verifying the single suite
        await manager.verify_single_shot(session_id, "adversarial", {})

        # Verify it is completed
        session = await manager.get_session(session_id)
        assert session is not None
        assert session["status"] == SessionStatus.COMPLETED.value

        # Attempting to cancel should return False
        result = await manager.cancel_session(session_id, "user1")
        assert result is False


# ---- 3. Cancel already cancelled session ----


class TestCancelCancelledSession:
    """Test that cancelling an already cancelled session returns False."""

    @pytest.mark.asyncio
    async def test_cancel_already_cancelled_returns_false(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        session_id, _, _ = await manager.create_session(
            user_id="user1", suites=["adversarial"]
        )

        # Cancel once - should succeed
        assert await manager.cancel_session(session_id, "user1") is True

        # Cancel again - should return False
        assert await manager.cancel_session(session_id, "user1") is False


# ---- 4. verify_single_shot on expired/nonexistent session ----


class TestVerifySingleShotNonexistent:
    """Test verify_single_shot raises ValueError for missing sessions."""

    @pytest.mark.asyncio
    async def test_verify_nonexistent_session_raises_not_found(
        self, manager: SessionManager
    ) -> None:
        with pytest.raises(ValueError, match="Session not found"):
            await manager.verify_single_shot("nonexistent-id", "adversarial", {})

    @pytest.mark.asyncio
    async def test_verify_expired_session_raises_not_found(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        session_id, _, _ = await manager.create_session(
            user_id="user1", suites=["adversarial"]
        )
        # Simulate expiry by removing the session from the store
        fake_redis._store.pop(_key(session_id), None)

        with pytest.raises(ValueError, match="Session not found"):
            await manager.verify_single_shot(session_id, "adversarial", {})


# ---- 5. verify_single_shot on completed session ----


class TestVerifySingleShotCompleted:
    """Test verify_single_shot raises ValueError about state on completed session."""

    @pytest.mark.asyncio
    async def test_verify_on_completed_session_raises_state_error(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        session_id, _, _ = await manager.create_session(
            user_id="user1", suites=["adversarial", "native"]
        )
        # Complete adversarial
        await manager.verify_single_shot(session_id, "adversarial", {})
        # Complete native - session becomes COMPLETED
        await manager.verify_single_shot(session_id, "native", {})

        session = await manager.get_session(session_id)
        assert session is not None
        assert session["status"] == SessionStatus.COMPLETED.value

        # Trying to verify another suite on a completed session should raise
        # We need to inject a new suite into the session to even get past the suite check,
        # so instead we directly set the session status to completed with remaining suites
        session["status"] = SessionStatus.COMPLETED.value
        session["suites"].append("social")
        session["suites_completed"] = ["adversarial", "native"]
        await fake_redis.setex(
            _key(session_id), ACTIVE_SESSION_TTL, json.dumps(session)
        )

        with pytest.raises(ValueError, match="not in verifiable state"):
            await manager.verify_single_shot(session_id, "social", {})

    @pytest.mark.asyncio
    async def test_verify_on_cancelled_session_raises_state_error(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        session_id, _, _ = await manager.create_session(
            user_id="user1", suites=["adversarial"]
        )
        await manager.cancel_session(session_id, "user1")

        with pytest.raises(ValueError, match="not in verifiable state"):
            await manager.verify_single_shot(session_id, "adversarial", {})


# ---- 6. submit_round_answer on session without novel-reasoning ----


class TestSubmitRoundNoNovelReasoning:
    """Test submit_round_answer raises ValueError when session lacks novel-reasoning."""

    @pytest.mark.asyncio
    async def test_submit_round_without_novel_reasoning_raises(
        self, manager: SessionManager
    ) -> None:
        session_id, _, _ = await manager.create_session(
            user_id="user1", suites=["adversarial"]
        )

        with pytest.raises(ValueError, match="does not include novel-reasoning"):
            await manager.submit_round_answer(session_id, 1, {})


# ---- 7. submit_round_answer on expired session answers ----


class TestSubmitRoundExpiredAnswers:
    """Test submit_round_answer raises ValueError when answers have expired."""

    @pytest.mark.asyncio
    async def test_submit_round_with_expired_answers_raises(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        session_id, _, _ = await manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="easy"
        )
        # Delete the answers key to simulate expiry
        answers_key = _key(session_id, "answers")
        fake_redis._store.pop(answers_key, None)

        with pytest.raises(ValueError, match="expired"):
            await manager.submit_round_answer(session_id, 1, {})


# ---- 8. submit_round_answer on completed session ----


class TestSubmitRoundCompletedSession:
    """Test submit_round_answer raises ValueError about state on completed session."""

    @pytest.mark.asyncio
    async def test_submit_round_on_completed_session_raises(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        session_id, _, _ = await manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="easy"
        )
        # Manually mark session as completed
        session = await manager.get_session(session_id)
        assert session is not None
        session["status"] = SessionStatus.COMPLETED.value
        await fake_redis.setex(
            _key(session_id), ACTIVE_SESSION_TTL, json.dumps(session)
        )

        with pytest.raises(ValueError, match="not in answerable state"):
            await manager.submit_round_answer(session_id, 1, {})

    @pytest.mark.asyncio
    async def test_submit_round_on_cancelled_session_raises(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        session_id, _, _ = await manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="easy"
        )
        await manager.cancel_session(session_id, "user1")

        with pytest.raises(ValueError, match="not in answerable state"):
            await manager.submit_round_answer(session_id, 1, {})

    @pytest.mark.asyncio
    async def test_submit_round_nonexistent_session_raises(
        self, manager: SessionManager
    ) -> None:
        with pytest.raises(ValueError, match="Session not found"):
            await manager.submit_round_answer("nonexistent-id", 1, {})


# ---- 9. submit_round_answer final round completes the suite ----


class TestSubmitRoundFinalRound:
    """Test that completing all rounds of novel-reasoning marks it as completed."""

    @pytest.mark.asyncio
    async def test_final_round_completes_novel_reasoning_only_session(
        self, manager: SessionManager
    ) -> None:
        session_id, challenges, _ = await manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="easy"
        )
        # easy difficulty has num_rounds=2
        novel_challenges = challenges.get("novel-reasoning", {}).get("challenges", {})
        round_answers: dict[str, Any] = {
            "challenges": {name: {"test_outputs": []} for name in novel_challenges}
        }

        # Submit round 1
        result1 = await manager.submit_round_answer(session_id, 1, round_answers)
        assert result1["next_round_data"] is not None
        assert result1["next_round_data"]["round"] == 2

        # Submit round 2 (final for easy)
        result2 = await manager.submit_round_answer(session_id, 2, round_answers)
        assert result2["next_round_data"] is None

        # Session should be completed since novel-reasoning is the only suite
        session = await manager.get_session(session_id)
        assert session is not None
        assert session["status"] == SessionStatus.COMPLETED.value
        assert MULTI_ROUND_SUITE in session["suites_completed"]
        assert MULTI_ROUND_SUITE in session["suite_results"]

    @pytest.mark.asyncio
    async def test_final_round_result_includes_iteration_curve(
        self, manager: SessionManager
    ) -> None:
        session_id, challenges, _ = await manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="easy"
        )
        novel_challenges = challenges.get("novel-reasoning", {}).get("challenges", {})
        round_answers: dict[str, Any] = {
            "challenges": {name: {"test_outputs": []} for name in novel_challenges}
        }

        await manager.submit_round_answer(session_id, 1, round_answers)
        await manager.submit_round_answer(session_id, 2, round_answers)

        feedback_r2 = await manager.get_round_feedback(session_id, 2)
        assert feedback_r2 is not None
        assert feedback_r2["round"] == 2

        session = await manager.get_session(session_id)
        assert session is not None
        novel_result = session["suite_results"][MULTI_ROUND_SUITE]
        assert "iteration_curve" in novel_result
        assert "passed" in novel_result
        assert "score" in novel_result
        assert "details" in novel_result
        assert "signature" in novel_result["details"]


# ---- 10. get_round_feedback for non-existent session ----


class TestGetRoundFeedbackNonExistent:
    """Test get_round_feedback returns None for sessions that do not exist."""

    @pytest.mark.asyncio
    async def test_feedback_nonexistent_session_returns_none(
        self, manager: SessionManager
    ) -> None:
        result = await manager.get_round_feedback("nonexistent-id", 1)
        assert result is None


# ---- 11. get_round_feedback for existing round ----


class TestGetRoundFeedbackExisting:
    """Test get_round_feedback returns round data after submission."""

    @pytest.mark.asyncio
    async def test_feedback_existing_round_returns_data(
        self, manager: SessionManager
    ) -> None:
        session_id, challenges, _ = await manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="easy"
        )
        novel_challenges = challenges.get("novel-reasoning", {}).get("challenges", {})
        round_answers: dict[str, Any] = {
            "challenges": {name: {"test_outputs": []} for name in novel_challenges}
        }

        await manager.submit_round_answer(session_id, 1, round_answers)

        feedback = await manager.get_round_feedback(session_id, 1)
        assert feedback is not None
        assert feedback["round"] == 1
        assert "accuracy" in feedback
        assert "response_time_ms" in feedback

    @pytest.mark.asyncio
    async def test_feedback_unsubmitted_round_returns_none(
        self, manager: SessionManager
    ) -> None:
        session_id, _, _ = await manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="easy"
        )
        # Round 1 not yet submitted
        feedback = await manager.get_round_feedback(session_id, 1)
        assert feedback is None

    @pytest.mark.asyncio
    async def test_round_timing_is_per_round_not_cumulative_session_age(
        self,
        manager: SessionManager,
        fake_redis: FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id, challenges, _ = await manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="easy"
        )
        session = await manager.get_session(session_id)
        assert session is not None
        session["start_time"] = 100.0
        session["round_started_at"] = 100.0
        session["time_budget_ms"] = 100_000
        await fake_redis.setex(
            _key(session_id), ACTIVE_SESSION_TTL, json.dumps(session)
        )
        timestamps = iter([101.0, 101.0, 101.0, 101.4, 101.4, 101.4, 101.4])
        monkeypatch.setattr(
            "mettle.session_manager.time.time", lambda: next(timestamps)
        )
        names = challenges["novel-reasoning"]["challenges"]
        answers: dict[str, Any] = {
            "challenges": {name: {"test_outputs": []} for name in names}
        }

        await manager.submit_round_answer(session_id, 1, answers)
        await manager.submit_round_answer(session_id, 2, answers)

        first = await manager.get_round_feedback(session_id, 1)
        second = await manager.get_round_feedback(session_id, 2)
        assert first is not None and second is not None
        assert first["response_time_ms"] == pytest.approx(1000.0)
        assert second["response_time_ms"] == pytest.approx(400.0)


# ---- 12. get_result with completed session and start_time ----


class TestGetResultWithStartTime:
    """Test get_result returns proper elapsed_ms when start_time is set."""

    @pytest.mark.asyncio
    async def test_result_has_positive_elapsed_ms(
        self, manager: SessionManager
    ) -> None:
        session_id, _, _ = await manager.create_session(
            user_id="user1", suites=["adversarial"]
        )
        # Verify to complete and set start_time
        await manager.verify_single_shot(session_id, "adversarial", {})

        result = await manager.get_result(session_id)
        assert result is not None
        assert result["elapsed_ms"] >= 0
        assert result["status"] == SessionStatus.COMPLETED.value
        assert result["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_result_nonexistent_session_returns_none(
        self, manager: SessionManager
    ) -> None:
        result = await manager.get_result("nonexistent-id")
        assert result is None


# ---- 13. get_result with empty suite results ----


class TestGetResultEmptySuiteResults:
    """Completed state with missing results is corruption, never a pass."""

    @pytest.mark.asyncio
    async def test_empty_results_fail_closed(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        # Inject a completed session with empty suite_results
        session_data: dict[str, Any] = {
            "session_id": "test-empty",
            "user_id": "user1",
            "entity_id": None,
            "suites": ["adversarial"],
            "difficulty": "standard",
            "status": SessionStatus.COMPLETED.value,
            "created_at": "2026-01-01T00:00:00+00:00",
            "expires_at": "2026-01-01T00:05:00+00:00",
            "time_budget_ms": 30000,
            "start_time": time.time() - 5,
            "current_round": 0,
            "suites_completed": ["adversarial"],
            "suite_results": {},
            "round_data": [],
        }
        await _inject_session(fake_redis, "test-empty", session_data)

        with pytest.raises(SessionStateError, match="result set"):
            await manager.get_result("test-empty")


# ---- 14. get_result with all suites passed ----


class TestGetResultAllPassed:
    """Test that all suites passing results in overall_passed=True."""

    @pytest.mark.asyncio
    async def test_all_suites_passed_overall_true(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        session_data = {
            "session_id": "test-allpass",
            "user_id": "user1",
            "entity_id": None,
            "suites": ["adversarial", "native"],
            "difficulty": "standard",
            "status": SessionStatus.COMPLETED.value,
            "created_at": "2026-01-01T00:00:00+00:00",
            "expires_at": "2026-01-01T00:05:00+00:00",
            "time_budget_ms": 60000,
            "start_time": time.time() - 10,
            "completed_at": time.time(),
            "current_round": 0,
            "suites_completed": ["adversarial", "native"],
            "suite_results": {
                "adversarial": {"passed": True, "score": 0.9},
                "native": {"passed": True, "score": 0.85},
            },
            "round_data": [],
        }
        await _inject_session(fake_redis, "test-allpass", session_data)

        result = await manager.get_result("test-allpass")
        assert result is not None
        assert result["overall_passed"] is True

    @pytest.mark.asyncio
    async def test_one_suite_failed_overall_false(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        session_data = {
            "session_id": "test-partialfail",
            "user_id": "user1",
            "entity_id": None,
            "suites": ["adversarial", "native"],
            "difficulty": "standard",
            "status": SessionStatus.COMPLETED.value,
            "created_at": "2026-01-01T00:00:00+00:00",
            "expires_at": "2026-01-01T00:05:00+00:00",
            "time_budget_ms": 60000,
            "start_time": time.time() - 10,
            "completed_at": time.time(),
            "current_round": 0,
            "suites_completed": ["adversarial", "native"],
            "suite_results": {
                "adversarial": {"passed": True, "score": 0.9},
                "native": {"passed": False, "score": 0.3},
            },
            "round_data": [],
        }
        await _inject_session(fake_redis, "test-partialfail", session_data)

        result = await manager.get_result("test-partialfail")
        assert result is not None
        assert result["overall_passed"] is False


# ---- 15. get_result with novel-reasoning results ----


class TestGetResultNovelReasoning:
    """Test get_result includes iteration_curve when novel-reasoning was run."""

    @pytest.mark.asyncio
    async def test_result_includes_iteration_curve(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        curve_data = {
            "time_trend": 0.8,
            "improvement": 0.5,
            "feedback_responsiveness": 0.6,
            "round1_suspicion": 0.0,
            "overall": 0.7,
            "signature": "AI",
        }
        session_data = {
            "session_id": "test-novel",
            "user_id": "user1",
            "entity_id": None,
            "suites": ["novel-reasoning"],
            "difficulty": "standard",
            "status": SessionStatus.COMPLETED.value,
            "created_at": "2026-01-01T00:00:00+00:00",
            "expires_at": "2026-01-01T00:05:00+00:00",
            "time_budget_ms": 120000,
            "start_time": time.time() - 20,
            "completed_at": time.time(),
            "current_round": 3,
            "suites_completed": ["novel-reasoning"],
            "suite_results": {
                "novel-reasoning": {
                    "passed": True,
                    "score": 0.7,
                    "iteration_curve": curve_data,
                    "round_data": [],
                    "details": {"signature": "AI", "threshold": 0.65},
                }
            },
            "round_data": [],
        }
        await _inject_session(fake_redis, "test-novel", session_data)

        result = await manager.get_result("test-novel")
        assert result is not None
        assert result["iteration_curve"] is not None
        assert result["iteration_curve"]["signature"] == "AI"
        assert result["iteration_curve"]["overall"] == 0.7

    @pytest.mark.asyncio
    async def test_result_without_novel_reasoning_has_no_curve(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        session_data = {
            "session_id": "test-nocurve",
            "user_id": "user1",
            "entity_id": None,
            "suites": ["adversarial"],
            "difficulty": "standard",
            "status": SessionStatus.COMPLETED.value,
            "created_at": "2026-01-01T00:00:00+00:00",
            "expires_at": "2026-01-01T00:05:00+00:00",
            "time_budget_ms": 30000,
            "start_time": time.time() - 5,
            "completed_at": time.time(),
            "current_round": 0,
            "suites_completed": ["adversarial"],
            "suite_results": {
                "adversarial": {"passed": True, "score": 0.9},
            },
            "round_data": [],
        }
        await _inject_session(fake_redis, "test-nocurve", session_data)

        result = await manager.get_result("test-nocurve")
        assert result is not None
        assert result["iteration_curve"] is None


# ---- 16. _analyze_iteration_curve ----


class TestAnalyzeIterationCurve:
    """Test _analyze_iteration_curve with known round data."""

    def test_analyze_with_improving_rounds(self, manager: SessionManager) -> None:
        round_data = [
            {"round": 1, "response_time_ms": 500.0, "accuracy": 0.3},
            {"round": 2, "response_time_ms": 400.0, "accuracy": 0.6},
            {"round": 3, "response_time_ms": 300.0, "accuracy": 0.85},
        ]
        server_answers = {"pass_threshold": 0.65}

        result = manager._analyze_iteration_curve(round_data, server_answers)

        assert "passed" in result
        assert "score" in result
        assert "iteration_curve" in result
        assert "round_data" in result
        assert "details" in result
        assert result["details"]["threshold"] == 0.65
        # With improving accuracy and decreasing time, should detect as AI
        assert result["iteration_curve"]["signature"] in ("AI", "HUMAN")
        assert isinstance(result["score"], float)

    def test_analyze_with_flat_rounds_detected_as_script(
        self, manager: SessionManager
    ) -> None:
        round_data = [
            {"round": 1, "response_time_ms": 100.0, "accuracy": 0.99},
            {"round": 2, "response_time_ms": 100.0, "accuracy": 0.99},
            {"round": 3, "response_time_ms": 100.0, "accuracy": 0.99},
        ]
        server_answers = {"pass_threshold": 0.65}

        result = manager._analyze_iteration_curve(round_data, server_answers)

        # Perfect from start with no variation should flag as SCRIPT
        assert result["iteration_curve"]["signature"] == "SCRIPT"
        # SCRIPT signature means passed should be False regardless of score
        assert result["passed"] is False

    def test_analyze_with_single_round_returns_script(
        self, manager: SessionManager
    ) -> None:
        round_data = [
            {"round": 1, "response_time_ms": 200.0, "accuracy": 0.5},
        ]
        server_answers = {"pass_threshold": 0.65}

        result = manager._analyze_iteration_curve(round_data, server_answers)
        # Fewer than 2 rounds returns SCRIPT and overall=0.0
        assert result["iteration_curve"]["signature"] == "SCRIPT"
        assert result["iteration_curve"]["overall"] == 0.0
        assert result["passed"] is False

    def test_analyze_uses_default_threshold_when_missing(
        self, manager: SessionManager
    ) -> None:
        round_data = [
            {"round": 1, "response_time_ms": 500.0, "accuracy": 0.3},
            {"round": 2, "response_time_ms": 300.0, "accuracy": 0.8},
        ]
        # No pass_threshold in server_answers
        server_answers: dict[str, Any] = {}

        result = manager._analyze_iteration_curve(round_data, server_answers)
        # Should default to 0.65
        assert result["details"]["threshold"] == 0.65

    def test_analyze_with_human_like_pattern(self, manager: SessionManager) -> None:
        round_data = [
            {"round": 1, "response_time_ms": 200.0, "accuracy": 0.4},
            {"round": 2, "response_time_ms": 350.0, "accuracy": 0.55},
            {"round": 3, "response_time_ms": 500.0, "accuracy": 0.65},
        ]
        server_answers = {"pass_threshold": 0.65}

        result = manager._analyze_iteration_curve(round_data, server_answers)
        # Increasing response time suggests HUMAN
        assert result["iteration_curve"]["signature"] == "HUMAN"


# ---- 19. Multiple suites with partial completion ----


class TestMultipleSuitesPartialCompletion:
    """Test that session stays IN_PROGRESS when not all suites are done."""

    @pytest.mark.asyncio
    async def test_session_stays_in_progress_with_remaining_suites(
        self, manager: SessionManager
    ) -> None:
        session_id, _, _ = await manager.create_session(
            user_id="user1", suites=["adversarial", "native", "social"]
        )

        # Complete only one suite
        await manager.verify_single_shot(session_id, "adversarial", {})

        session = await manager.get_session(session_id)
        assert session is not None
        assert session["status"] == SessionStatus.IN_PROGRESS.value
        assert "adversarial" in session["suites_completed"]
        assert len(session["suites_completed"]) == 1

    @pytest.mark.asyncio
    async def test_session_completes_after_all_suites_done(
        self, manager: SessionManager
    ) -> None:
        session_id, _, _ = await manager.create_session(
            user_id="user1", suites=["adversarial", "native"]
        )

        await manager.verify_single_shot(session_id, "adversarial", {})
        session_mid = await manager.get_session(session_id)
        assert session_mid is not None
        assert session_mid["status"] == SessionStatus.IN_PROGRESS.value

        await manager.verify_single_shot(session_id, "native", {})
        session_done = await manager.get_session(session_id)
        assert session_done is not None
        assert session_done["status"] == SessionStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_novel_reasoning_final_round_with_remaining_single_shot_stays_in_progress(
        self, manager: SessionManager
    ) -> None:
        session_id, challenges, _ = await manager.create_session(
            user_id="user1",
            suites=["adversarial", "novel-reasoning"],
            difficulty="easy",
        )
        # Complete novel-reasoning (2 rounds for easy)
        novel_challenges = challenges.get("novel-reasoning", {}).get("challenges", {})
        round_answers: dict[str, Any] = {
            "challenges": {name: {"test_outputs": []} for name in novel_challenges}
        }

        await manager.submit_round_answer(session_id, 1, round_answers)
        await manager.submit_round_answer(session_id, 2, round_answers)

        session = await manager.get_session(session_id)
        assert session is not None
        # adversarial still not done, so should be IN_PROGRESS
        assert session["status"] == SessionStatus.IN_PROGRESS.value
        assert MULTI_ROUND_SUITE in session["suites_completed"]
        assert "adversarial" not in session["suites_completed"]


# ---- 20. Time budget calculation ----


class TestTimeBudgetCalculation:
    """Test time budget with and without novel reasoning."""

    @pytest.mark.asyncio
    async def test_time_budget_single_shot_only(self, manager: SessionManager) -> None:
        _, _, meta = await manager.create_session(
            user_id="user1", suites=["adversarial", "native"]
        )
        # 2 single-shot suites * 30000ms each = 60000ms
        assert meta["time_budget_ms"] == 60000

    @pytest.mark.asyncio
    async def test_time_budget_single_suite(self, manager: SessionManager) -> None:
        _, _, meta = await manager.create_session(
            user_id="user1", suites=["adversarial"]
        )
        assert meta["time_budget_ms"] == 30000

    @pytest.mark.asyncio
    async def test_time_budget_with_novel_reasoning_easy(
        self, manager: SessionManager
    ) -> None:
        _, _, meta = await manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="easy"
        )
        assert meta["time_budget_ms"] == 45_000

    @pytest.mark.asyncio
    async def test_time_budget_with_novel_reasoning_standard(
        self, manager: SessionManager
    ) -> None:
        _, _, meta = await manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="standard"
        )
        assert meta["time_budget_ms"] == 30_000

    @pytest.mark.asyncio
    async def test_time_budget_with_novel_reasoning_hard(
        self, manager: SessionManager
    ) -> None:
        _, _, meta = await manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="hard"
        )
        assert meta["time_budget_ms"] == 20_000

    @pytest.mark.asyncio
    async def test_time_budget_all_suites_easy(self, manager: SessionManager) -> None:
        _, _, meta = await manager.create_session(
            user_id="user1", suites=["all"], difficulty="easy"
        )
        # Eleven available suites: ten single-shot budgets plus novel once.
        assert meta["time_budget_ms"] == 45_000 + 10 * 30_000

    @pytest.mark.asyncio
    async def test_time_budget_mixed_suites_with_novel(
        self, manager: SessionManager
    ) -> None:
        _, _, meta = await manager.create_session(
            user_id="user1",
            suites=["adversarial", "native", "novel-reasoning"],
            difficulty="standard",
        )
        assert meta["time_budget_ms"] == 30_000 + 2 * 30_000


# ---- Additional edge cases ----


class TestSessionManagerEdgeCases:
    """Additional coverage for edge paths in session_manager.py."""

    @pytest.mark.asyncio
    async def test_get_session_answers_nonexistent(
        self, manager: SessionManager
    ) -> None:
        result = await manager.get_session_answers("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_session_answers_returns_dict(
        self, manager: SessionManager
    ) -> None:
        session_id, _, _ = await manager.create_session(
            user_id="user1", suites=["adversarial"]
        )
        answers = await manager.get_session_answers(session_id)
        assert answers is not None
        assert isinstance(answers, dict)
        assert "adversarial" in answers

    @pytest.mark.asyncio
    async def test_verify_uses_server_issue_time(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        session_id, _, _ = await manager.create_session(
            user_id="user1", suites=["adversarial", "native"]
        )
        session_before = await manager.get_session(session_id)
        assert session_before is not None
        issued_at = session_before["start_time"]
        assert isinstance(issued_at, float)

        await manager.verify_single_shot(session_id, "adversarial", {})
        session_after = await manager.get_session(session_id)
        assert session_after is not None
        assert session_after["start_time"] == issued_at

    @pytest.mark.asyncio
    async def test_submit_round_uses_server_issue_time(
        self, manager: SessionManager
    ) -> None:
        session_id, challenges, _ = await manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="easy"
        )
        session_before = await manager.get_session(session_id)
        assert session_before is not None
        issued_at = session_before["start_time"]
        assert isinstance(issued_at, float)

        novel_challenges = challenges.get("novel-reasoning", {}).get("challenges", {})
        round_answers: dict[str, Any] = {
            "challenges": {name: {"test_outputs": []} for name in novel_challenges}
        }
        await manager.submit_round_answer(session_id, 1, round_answers)

        session_after = await manager.get_session(session_id)
        assert session_after is not None
        assert session_after["start_time"] == issued_at

    @pytest.mark.asyncio
    async def test_submit_round_tracks_current_round(
        self, manager: SessionManager
    ) -> None:
        session_id, challenges, _ = await manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="easy"
        )
        novel_challenges = challenges.get("novel-reasoning", {}).get("challenges", {})
        round_answers: dict[str, Any] = {
            "challenges": {name: {"test_outputs": []} for name in novel_challenges}
        }

        await manager.submit_round_answer(session_id, 1, round_answers)
        session = await manager.get_session(session_id)
        assert session is not None
        assert session["current_round"] == 1

        await manager.submit_round_answer(session_id, 2, round_answers)
        session = await manager.get_session(session_id)
        assert session is not None
        assert session["current_round"] == 2

    @pytest.mark.asyncio
    async def test_submit_round_returns_time_remaining(
        self, manager: SessionManager
    ) -> None:
        session_id, challenges, _ = await manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="easy"
        )
        novel_challenges = challenges.get("novel-reasoning", {}).get("challenges", {})
        round_answers: dict[str, Any] = {
            "challenges": {name: {"test_outputs": []} for name in novel_challenges}
        }

        result = await manager.submit_round_answer(session_id, 1, round_answers)
        assert "time_remaining_ms" in result
        assert result["time_remaining_ms"] >= 0

    @pytest.mark.asyncio
    async def test_submit_round_returns_errors_list(
        self, manager: SessionManager
    ) -> None:
        session_id, challenges, _ = await manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="easy"
        )
        novel_challenges = challenges.get("novel-reasoning", {}).get("challenges", {})
        round_answers: dict[str, Any] = {
            "challenges": {name: {"test_outputs": []} for name in novel_challenges}
        }

        result = await manager.submit_round_answer(session_id, 1, round_answers)
        assert "errors" in result
        assert isinstance(result["errors"], list)
        # Errors list should be capped at 10
        assert len(result["errors"]) <= 10

    @pytest.mark.asyncio
    async def test_get_result_with_no_start_time_fails_closed(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        # Inject a completed session with start_time=None
        session_data = {
            "session_id": "test-nostart",
            "user_id": "user1",
            "entity_id": None,
            "suites": ["adversarial"],
            "difficulty": "standard",
            "status": SessionStatus.COMPLETED.value,
            "created_at": "2026-01-01T00:00:00+00:00",
            "expires_at": "2026-01-01T00:05:00+00:00",
            "time_budget_ms": 30000,
            "start_time": None,
            "current_round": 0,
            "suites_completed": ["adversarial"],
            "suite_results": {"adversarial": {"passed": True, "score": 0.9}},
            "round_data": [],
        }
        await _inject_session(fake_redis, "test-nostart", session_data)

        with pytest.raises(SessionStateError, match="start_time"):
            await manager.get_result("test-nostart")

    @pytest.mark.asyncio
    async def test_resolve_suites_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown suites"):
            SessionManager._resolve_suites(["not-a-suite"])

    @pytest.mark.asyncio
    async def test_resolve_suites_multiple_unknown_lists_all(self) -> None:
        with pytest.raises(ValueError, match="Unknown suites.*fake1.*fake2"):
            SessionManager._resolve_suites(["fake1", "fake2"])

    @pytest.mark.asyncio
    async def test_cancel_removes_from_active_set(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        session_id, _, _ = await manager.create_session(
            user_id="user1", suites=["adversarial"]
        )
        active_key = _rate_key("user1", "active")

        # Should be in active set
        assert session_id in fake_redis._sets.get(active_key, set())

        # Cancel
        await manager.cancel_session(session_id, "user1")

        # Should be removed from active set
        assert session_id not in fake_redis._sets.get(active_key, set())

    @pytest.mark.asyncio
    async def test_session_metadata_fields(self, manager: SessionManager) -> None:
        session_id, _, meta = await manager.create_session(
            user_id="user1", suites=["adversarial"], entity_id="ent-1"
        )
        assert meta["session_id"] == session_id
        assert meta["user_id"] == "user1"
        assert meta["entity_id"] == "ent-1"
        assert meta["status"] == SessionStatus.CHALLENGES_GENERATED.value
        assert isinstance(meta["start_time"], float)
        assert meta["current_round"] == 0
        assert meta["suites_completed"] == []
        assert meta["suite_results"] == {}
        assert meta["round_data"] == []
        assert "created_at" in meta
        assert "expires_at" in meta

    @pytest.mark.asyncio
    async def test_unknown_suite_in_generators_raises(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        # This tests the ValueError("Unknown suite: {suite}") path by injecting
        # a valid suite name that bypasses _resolve_suites but is not in generators.
        # Since SUITE_NAMES includes all known suites, the only way to reach this
        # is if _resolve_suites is bypassed. The code path is covered by the
        # existing test_create_session_invalid_suite in test_mettle_api.py,
        # but we verify the resolve_suites check catches it first.
        with pytest.raises(ValueError, match="Unknown suites"):
            await manager.create_session(user_id="user1", suites=["made-up-suite"])

        assert await fake_redis.scard(_rate_key("user1", "active")) == 0
        assert await fake_redis.get(_rate_key("user1", "hourly")) is None

    @pytest.mark.asyncio
    async def test_generation_failure_releases_reserved_quota(
        self,
        manager: SessionManager,
        fake_redis: FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail_generation():
            raise RuntimeError("generation failed")

        monkeypatch.setattr(
            "mettle.session_manager.ChallengeAdapter.generate_adversarial",
            fail_generation,
        )

        with pytest.raises(RuntimeError, match="generation failed"):
            await manager.create_session(user_id="user1", suites=["adversarial"])

        assert await fake_redis.scard(_rate_key("user1", "active")) == 0
        assert await fake_redis.get(_rate_key("user1", "hourly")) == "0"

    @pytest.mark.asyncio
    async def test_submit_round_with_flat_answers_dict(
        self, manager: SessionManager
    ) -> None:
        """Test that answers without a 'challenges' wrapper are handled."""
        session_id, challenges, _ = await manager.create_session(
            user_id="user1", suites=["novel-reasoning"], difficulty="easy"
        )
        novel_challenges = challenges.get("novel-reasoning", {}).get("challenges", {})
        # Submit answers as a flat dict (no 'challenges' key)
        flat_answers: dict[str, Any] = {
            name: {"test_outputs": []} for name in novel_challenges
        }

        result = await manager.submit_round_answer(session_id, 1, flat_answers)
        assert "accuracy" in result
        assert "round_num" in result
        assert result["round_num"] == 1


def _valid_session_record(
    session_id: str = "semantic-session",
    *,
    status: str = SessionStatus.IN_PROGRESS.value,
) -> dict[str, Any]:
    now = time.time()
    return {
        "session_id": session_id,
        "user_id": "semantic-user",
        "entity_id": None,
        "vcp_token": None,
        "operator_commitment": None,
        "suites": ["adversarial"],
        "difficulty": "standard",
        "status": status,
        "created_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "expires_at": datetime.fromtimestamp(now + 30, tz=timezone.utc).isoformat(),
        "time_budget_ms": 30_000,
        "start_time": now,
        "novel_started_at": None,
        "round_started_at": None,
        "current_round": 0,
        "suites_completed": [],
        "suite_results": {},
        "round_data": [],
        "presence": None,
    }


class TestAdversarialSessionStateRegressions:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw", ["{", "[]", "null", "1", '"text"'])
    async def test_malformed_or_nonobject_session_state_fails_closed(
        self, raw: str, fake_redis: FakeRedis
    ) -> None:
        fake_redis._store[_key("corrupt")] = raw

        with pytest.raises(SessionStateError, match="Corrupt session metadata"):
            await SessionManager(fake_redis).get_session("corrupt")

    @pytest.mark.asyncio
    async def test_completed_state_missing_results_fails_closed(
        self, fake_redis: FakeRedis
    ) -> None:
        session = _valid_session_record("incomplete", status="completed")
        session["suites_completed"] = ["adversarial"]
        session["completed_at"] = time.time()
        await fake_redis.setex(_key("incomplete"), 60, json.dumps(session))

        with pytest.raises(SessionStateError, match="result set"):
            await SessionManager(fake_redis).get_result("incomplete")

    @pytest.mark.asyncio
    async def test_completed_elapsed_is_stable_across_reads(
        self, fake_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = _valid_session_record("stable", status="completed")
        session.update(
            {
                "start_time": 10.0,
                "completed_at": 15.25,
                "suites_completed": ["adversarial"],
                "suite_results": {"adversarial": {"passed": True}},
            }
        )
        await fake_redis.setex(_key("stable"), 60, json.dumps(session))
        manager = SessionManager(fake_redis)

        monkeypatch.setattr("mettle.session_manager.time.time", lambda: 100.0)
        first = await manager.get_result("stable")
        monkeypatch.setattr("mettle.session_manager.time.time", lambda: 10_000.0)
        second = await manager.get_result("stable")

        assert first is not None and second is not None
        assert first["elapsed_ms"] == second["elapsed_ms"] == 5250

    @pytest.mark.asyncio
    async def test_expiry_uses_atomic_terminal_cleanup(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        session_id, _, session = await manager.create_session(
            "expiry-user", ["adversarial"]
        )
        session["start_time"] = time.time() - 60
        session["time_budget_ms"] = 1
        await fake_redis.setex(_key(session_id), 30, json.dumps(session))

        with pytest.raises(ValueError, match="time budget exceeded"):
            await manager.verify_single_shot(session_id, "adversarial", {})

        stored = await manager.get_session(session_id)
        assert stored is not None and stored["status"] == "expired"
        assert await fake_redis.get(_key(session_id, "answers")) is None
        assert session_id not in fake_redis._sets[_rate_key("expiry-user", "active")]

    @pytest.mark.asyncio
    async def test_stale_active_reservations_are_pruned_and_construction_ttl_is_safe(
        self, manager: SessionManager, fake_redis: FakeRedis
    ) -> None:
        active_key = _rate_key("prune-user", "active")
        fake_redis._sets[active_key] = {
            "stale-1",
            "stale-2",
            "stale-3",
            "stale-4",
            "stale-5",
        }

        session_id, _, session = await manager.create_session(
            "prune-user", ["adversarial"]
        )

        assert fake_redis._sets[active_key] == {session_id}
        assert fake_redis._ttls[active_key] == ACTIVE_SESSION_TTL
        assert session["time_budget_ms"] == 30_000

    @pytest.mark.asyncio
    async def test_generation_time_is_not_charged_to_respondent(
        self,
        manager: SessionManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        clock = [100.0]
        original = session_module.ChallengeAdapter.generate_adversarial

        def delayed_generation():
            clock[0] += 12.0
            return original()

        monkeypatch.setattr(
            "mettle.session_manager.ChallengeAdapter.generate_adversarial",
            delayed_generation,
        )
        monkeypatch.setattr("mettle.session_manager.time.time", lambda: clock[0])

        _, _, session = await manager.create_session("clock-user", ["adversarial"])

        assert session["start_time"] == 112.0
        assert datetime.fromisoformat(session["expires_at"]).timestamp() == 142.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("time_budget_s", True),
            ("num_rounds", 0),
            ("challenges", []),
        ],
    )
    async def test_corrupt_novel_round_state_fails_closed(
        self,
        manager: SessionManager,
        fake_redis: FakeRedis,
        field: str,
        value: Any,
    ) -> None:
        session_id, challenges, _ = await manager.create_session(
            "corrupt-novel-user", [MULTI_ROUND_SUITE], difficulty="easy"
        )
        challenge_names = set(challenges[MULTI_ROUND_SUITE]["challenges"])
        raw = await fake_redis.get(_key(session_id, "answers"))
        assert raw is not None
        server_answers = json.loads(raw)
        server_answers[MULTI_ROUND_SUITE][field] = value
        await fake_redis.setex(
            _key(session_id, "answers"), 45, json.dumps(server_answers)
        )
        submitted: dict[str, Any] = {
            "challenges": {name: {} for name in challenge_names}
        }

        with pytest.raises(SessionStateError, match="novel-reasoning answers"):
            await manager.submit_round_answer(session_id, 1, submitted)

    def test_corrupt_progressive_schedule_fails_closed(self) -> None:
        server = {
            "num_rounds": 3,
            "challenges": {
                "sequence_alchemy": {"round_data": {1: {}, 3: {}}},
            },
        }

        with pytest.raises(SessionStateError, match="round state"):
            SessionManager._next_novel_round_data(2, server)


class TestRedisLeaseAndAtomicityRegressions:
    @pytest.mark.asyncio
    async def test_long_transition_renews_token_owned_lease(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        redis = SemanticRedis()
        manager = SessionManager(redis)
        # Give the event loop ample scheduling margin.  The assertion targets
        # renewal activity itself; stale-owner fencing is covered separately.
        monkeypatch.setattr(session_module, "SESSION_LOCK_TTL", 1.0)
        monkeypatch.setattr(session_module, "SESSION_LOCK_RENEW_INTERVAL", 0.005)

        async with manager._session_lock("renewed"):
            for _ in range(100):
                if redis.renewals >= 2:
                    break
                await asyncio.sleep(0.005)

        assert redis.renewals >= 2

    @pytest.mark.asyncio
    async def test_concurrent_semantic_lock_has_single_owner(self) -> None:
        redis = SemanticRedis()
        manager = SessionManager(redis)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def owner() -> None:
            async with manager._session_lock("contended"):
                entered.set()
                await release.wait()

        task = asyncio.create_task(owner())
        await entered.wait()
        with pytest.raises(ValueError, match="already in progress"):
            async with manager._session_lock("contended"):
                pytest.fail("second owner entered")
        release.set()
        await task

    @pytest.mark.asyncio
    async def test_lease_loss_prevents_state_commit(self) -> None:
        redis = SemanticRedis()
        manager = SessionManager(redis)
        session = _valid_session_record("lost")
        await redis.setex(_key("lost"), 30, json.dumps(session))
        original = await redis.get(_key("lost"))

        with pytest.raises(SessionLockLostError):
            async with manager._session_lock("lost"):
                redis.replace_lock_owner("lost")
                session["current_round"] = 1
                await manager._persist_active_session(session, 30)

        assert await redis.get(_key("lost")) == original

    @pytest.mark.asyncio
    async def test_terminal_transition_is_atomic_and_deletes_answers(self) -> None:
        redis = SemanticRedis()
        manager = SessionManager(redis)
        session = _valid_session_record("terminal")
        await redis.setex(_key("terminal"), 30, json.dumps(session))
        await redis.setex(_key("terminal", "answers"), 30, json.dumps({"x": 1}))
        await redis.sadd(_rate_key("semantic-user", "active"), "terminal")
        session.update(
            {
                "status": "completed",
                "completed_at": time.time(),
                "suites_completed": ["adversarial"],
                "suite_results": {"adversarial": {"passed": True}},
            }
        )

        async with manager._session_lock("terminal"):
            await manager._persist_terminal_session(session)

        assert await redis.get(_key("terminal", "answers")) is None
        assert await redis.scard(_rate_key("semantic-user", "active")) == 0
        stored = await manager.get_session("terminal")
        assert stored is not None and stored["status"] == "completed"

    @pytest.mark.asyncio
    async def test_atomic_terminal_failure_leaves_all_state_unchanged(self) -> None:
        redis = SemanticRedis()
        manager = SessionManager(redis)
        session = _valid_session_record("terminal-fault")
        original = json.dumps(session)
        answers = json.dumps({"secret": 1})
        await redis.setex(_key("terminal-fault"), 30, original)
        await redis.setex(_key("terminal-fault", "answers"), 30, answers)
        await redis.sadd(_rate_key("semantic-user", "active"), "terminal-fault")
        session["status"] = "cancelled"
        session["terminated_at"] = time.time()
        redis.fail_terminal = True

        with pytest.raises(RedisError, match="injected atomic terminal failure"):
            async with manager._session_lock("terminal-fault"):
                await manager._persist_terminal_session(session)

        assert await redis.get(_key("terminal-fault")) == original
        assert await redis.get(_key("terminal-fault", "answers")) == answers
        assert await redis.scard(_rate_key("semantic-user", "active")) == 1

    @pytest.mark.asyncio
    async def test_corrupt_active_index_cannot_partially_commit_terminal_state(
        self,
    ) -> None:
        redis = SemanticRedis()
        manager = SessionManager(redis)
        session = _valid_session_record("terminal-corrupt-index")
        original = json.dumps(session)
        answers = json.dumps({"secret": 1})
        await redis.setex(_key("terminal-corrupt-index"), 30, original)
        await redis.setex(_key("terminal-corrupt-index", "answers"), 30, answers)
        await redis.sadd(_rate_key("semantic-user", "active"), "terminal-corrupt-index")
        session["status"] = "cancelled"
        session["terminated_at"] = time.time()
        redis.corrupt_active_index = True

        with pytest.raises(SessionStateError, match="active-session index"):
            async with manager._session_lock("terminal-corrupt-index"):
                await manager._persist_terminal_session(session)

        assert await redis.get(_key("terminal-corrupt-index")) == original
        assert await redis.get(_key("terminal-corrupt-index", "answers")) == answers
        assert await redis.scard(_rate_key("semantic-user", "active")) == 1


class TestPresentationRaceAndBoundaryRegressions:
    @pytest.mark.asyncio
    async def test_sliding_window_has_no_fixed_minute_double_burst(
        self, fake_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = SessionManager(fake_redis)
        monkeypatch.setattr(session_module, "MAX_PRESENTATION_CHALLENGES_PER_MINUTE", 2)
        clock = [59.999]
        monkeypatch.setattr("mettle.session_manager.time.time", lambda: clock[0])
        kwargs = {
            "verifier_user_id": "boundary-user",
            "credential_jti": "jti",
            "audience": "audience",
        }

        await manager.create_presentation_challenge(**kwargs)
        await manager.create_presentation_challenge(**kwargs)
        clock[0] = 60.001
        with pytest.raises(SessionRateLimitError):
            await manager.create_presentation_challenge(**kwargs)
        clock[0] = 119.999
        await manager.create_presentation_challenge(**kwargs)

        state = json.loads(fake_redis._store[_presentation_rate_key("boundary-user")])
        assert len(state["timestamps"]) == 1

    @pytest.mark.asyncio
    async def test_concurrent_presentation_consumes_exactly_one(
        self, fake_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager_a = SessionManager(fake_redis)
        manager_b = SessionManager(fake_redis)
        challenge_id = "single-use"
        challenge = {
            "challenge_id": challenge_id,
            "nonce": "nonce",
            "audience": "audience",
            "credential_jti": "jti",
            "verifier_user_id": "verifier",
            "expires_at": "2099-01-01T00:00:00+00:00",
        }
        await fake_redis.setex(
            _presentation_key(challenge_id), 60, json.dumps(challenge)
        )
        monkeypatch.setattr(
            "mettle.session_manager.verify_holder_signature", lambda **_kwargs: None
        )
        kwargs = {
            "verifier_user_id": "verifier",
            "challenge_id": challenge_id,
            "credential_jti": "jti",
            "audience": "audience",
            "public_key_pem": "unused",
            "holder_signature": "unused",
        }

        outcomes = await asyncio.gather(
            manager_a.verify_presentation(**kwargs),
            manager_b.verify_presentation(**kwargs),
            return_exceptions=True,
        )

        assert sum(isinstance(outcome, dict) for outcome in outcomes) == 1
        assert sum(isinstance(outcome, ValueError) for outcome in outcomes) == 1
        assert await fake_redis.get(_presentation_key(challenge_id)) is None
