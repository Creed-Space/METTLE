"""Session manager for METTLE API.

Handles session lifecycle, Redis storage, timing enforcement, and rate limiting.
Sessions follow the state machine: CREATED -> CHALLENGES_GENERATED -> IN_PROGRESS -> COMPLETED/EXPIRED/CANCELLED
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import secrets
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from redis.exceptions import RedisError

from mettle.api_models import (
    GOVERNANCE_SUITE,
    MULTI_ROUND_SUITE,
    SUITE_NAMES,
    SessionStatus,
)
from mettle.challenge_adapter import ChallengeAdapter
from mettle.continuity import (
    CONTINUITY_ANSWER_KEY,
    attach_continuity_challenge,
    retire_continuity_secret,
    verify_continuity_answer,
)
from mettle.app_config import settings
from mettle.llm_challenges import is_available as llm_available
from mettle.presence import (
    advance_session_presence,
    issuer_signed_session_presence,
    new_session_presence,
    verify_holder_signature,
    verify_submission_proof,
)

LLM_DYNAMIC_SUITE = "llm-dynamic"

logger = logging.getLogger(__name__)

# TTLs in seconds
ACTIVE_SESSION_TTL = 300  # 5 minutes
COMPLETED_SESSION_TTL = 3600  # 1 hour
RATE_LIMIT_WINDOW = 3600  # 1 hour

# Rate limits
MAX_ACTIVE_SESSIONS_PER_USER = 5
MAX_SESSIONS_PER_HOUR = 100
SESSION_LOCK_TTL = 30
SESSION_LOCK_RENEW_INTERVAL = SESSION_LOCK_TTL / 3
PRESENTATION_CHALLENGE_TTL = 60
MAX_PRESENTATION_CHALLENGES_PER_MINUTE = 60
PRESENTATION_RATE_WINDOW_MS = 60_000


class SessionRateLimitError(ValueError):
    """Session creation quota exceeded, with an HTTP retry hint."""

    def __init__(self, message: str, retry_after: int) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class SessionStateError(RedisError):
    """Stored session state is corrupt or violates a lifecycle invariant."""


class SessionLockLostError(RedisError):
    """A worker lost its Redis lease while changing session state."""


@dataclass
class _SessionLease:
    """Token-owned lease held by the current session transition."""

    lock_key: str
    token: str | None
    lost: asyncio.Event = field(default_factory=asyncio.Event)
    renewal_error: BaseException | None = None


_CURRENT_SESSION_LEASE: ContextVar[_SessionLease | None] = ContextVar(
    "mettle_current_session_lease", default=None
)


_RATE_RESERVATION_SCRIPT = """
local members = redis.call('SMEMBERS', KEYS[1])
for _, session_id in ipairs(members) do
  local session_key = ARGV[6] .. session_id
  local reservation_key = session_key .. ':reservation'
  if redis.call('EXISTS', session_key) == 0 and redis.call('EXISTS', reservation_key) == 0 then
    redis.call('SREM', KEYS[1], session_id)
  end
end
local active_count = redis.call('SCARD', KEYS[1])
if active_count >= tonumber(ARGV[2]) then return -1 end
local hourly_count = tonumber(redis.call('GET', KEYS[2]) or '0')
if hourly_count >= tonumber(ARGV[3]) then return -2 end
redis.call('SET', KEYS[3], '1', 'EX', tonumber(ARGV[4]))
redis.call('SADD', KEYS[1], ARGV[1])
local active_ttl = redis.call('TTL', KEYS[1])
if active_ttl < tonumber(ARGV[4]) then
  redis.call('EXPIRE', KEYS[1], tonumber(ARGV[4]))
end
hourly_count = redis.call('INCR', KEYS[2])
if hourly_count == 1 then
  redis.call('EXPIRE', KEYS[2], tonumber(ARGV[5]))
end
return hourly_count
"""

_LOCK_RELEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""

_LOCK_RENEW_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
end
return 0
"""

_LOCK_CHECK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then return 1 end
return 0
"""

_ACTIVE_PERSIST_SCRIPT = """
if redis.call('GET', KEYS[2]) ~= ARGV[3] then return -1 end
redis.call('SET', KEYS[1], ARGV[1], 'EX', tonumber(ARGV[2]))
return 1
"""

_TERMINAL_PERSIST_SCRIPT = """
if redis.call('GET', KEYS[4]) ~= ARGV[4] then return -1 end
local active_type = redis.call('TYPE', KEYS[2]).ok
if active_type ~= 'none' and active_type ~= 'set' then return -2 end
redis.call('SET', KEYS[1], ARGV[1], 'EX', tonumber(ARGV[2]))
redis.call('SREM', KEYS[2], ARGV[3])
redis.call('DEL', KEYS[3])
redis.call('DEL', KEYS[5])
return 1
"""

_RATE_RESERVATION_RELEASE_SCRIPT = """
local removed = redis.call('SREM', KEYS[1], ARGV[1])
redis.call('DEL', KEYS[3])
if removed == 1 then
  local hourly_count = tonumber(redis.call('GET', KEYS[2]) or '0')
  if hourly_count > 0 then redis.call('DECR', KEYS[2]) end
end
return removed
"""

_PRESENTATION_RATE_SCRIPT = """
local now_ms = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now_ms - window_ms)
local count = redis.call('ZCARD', KEYS[1])
if count >= limit then
  local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
  local retry_ms = window_ms
  if oldest[2] then retry_ms = math.max(1, tonumber(oldest[2]) + window_ms - now_ms) end
  return {-1, retry_ms}
end
redis.call('ZADD', KEYS[1], now_ms, ARGV[4])
redis.call('PEXPIRE', KEYS[1], window_ms)
return {count + 1, 0}
"""

_PRESENTATION_CONSUME_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


def _key(session_id: str, suffix: str = "") -> str:
    """Build a Redis key."""
    base = f"{settings.redis_namespace}:session:{session_id}"
    return f"{base}:{suffix}" if suffix else base


def _rate_key(user_id: str, kind: str) -> str:
    return f"{settings.redis_namespace}:rate:{user_id}:{kind}"


def _presentation_key(challenge_id: str) -> str:
    return f"{settings.redis_namespace}:presentation:{challenge_id}"


def _presentation_rate_key(user_id: str) -> str:
    return f"{settings.redis_namespace}:presentation-rate:{user_id}"


class SessionManager:
    """Manages METTLE verification sessions backed by Redis."""

    def __init__(self, redis_client: Any) -> None:
        self.redis = redis_client
        locks = getattr(redis_client, "_mettle_local_locks", None)
        if not isinstance(locks, dict):
            locks = {}
            try:
                setattr(redis_client, "_mettle_local_locks", locks)
            except (AttributeError, TypeError):
                pass
        self._local_locks: dict[str, asyncio.Lock] = locks

    # ---- Session Lifecycle ----

    async def create_session(
        self,
        user_id: str,
        suites: list[str],
        difficulty: str = "standard",
        entity_id: str | None = None,
        vcp_token: str | None = None,
        operator_commitment: dict[str, Any] | None = None,
        presence: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Create a new verification session.

        Returns (session_id, client_challenges, session_metadata).
        Raises ValueError on rate limit or invalid suites.
        """
        # Reject malformed or unavailable suite requests before reserving quota.
        # Otherwise five invalid requests can occupy every active-session slot.
        resolved_suites = self._resolve_suites(suites)
        if LLM_DYNAMIC_SUITE in resolved_suites and not llm_available():
            raise ValueError(
                "llm-dynamic suite requires ANTHROPIC_API_KEY and anthropic package"
            )

        time_budget_ms = self._session_time_budget_ms(resolved_suites, difficulty)
        storage_ttl = max(1, math.ceil(time_budget_ms / 1000))
        # A short respondent budget must not also become a shorter construction
        # lease. The LLM generator alone may legitimately consume its entire
        # 30-second call timeout before the session keys exist. Keep the
        # reservation alive for the established construction window, then
        # delete it immediately after successful persistence below.
        construction_ttl = max(storage_ttl, ACTIVE_SESSION_TTL)

        # Reserve both active and hourly quota atomically before doing expensive
        # challenge generation. Concurrent requests cannot all pass a separate
        # check and increment later. Any generation/storage failure releases the
        # reservation below.
        session_id = secrets.token_urlsafe(32)
        await self._reserve_rate_limits(
            user_id,
            session_id,
            active_ttl=construction_ttl,
        )

        try:
            return await self._create_reserved_session(
                user_id=user_id,
                session_id=session_id,
                resolved_suites=resolved_suites,
                difficulty=difficulty,
                entity_id=entity_id,
                vcp_token=vcp_token,
                operator_commitment=operator_commitment,
                presence_registration=presence,
                time_budget_ms=time_budget_ms,
                storage_ttl=storage_ttl,
            )
        except Exception:
            await self._release_rate_reservation(user_id, session_id)
            raise

    async def _create_reserved_session(
        self,
        *,
        user_id: str,
        session_id: str,
        resolved_suites: list[str],
        difficulty: str,
        entity_id: str | None,
        vcp_token: str | None,
        operator_commitment: dict[str, Any] | None,
        presence_registration: dict[str, Any] | None,
        time_budget_ms: int,
        storage_ttl: int,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Generate and persist a session after its Redis quota reservation."""
        # Generate challenges for each suite
        client_challenges: dict[str, Any] = {}
        server_answers: dict[str, Any] = {}

        generators: dict[str, Any] = {
            "adversarial": ChallengeAdapter.generate_adversarial,
            "native": ChallengeAdapter.generate_native,
            "self-reference": ChallengeAdapter.generate_self_reference,
            "social": ChallengeAdapter.generate_social,
            "inverse-turing": ChallengeAdapter.generate_inverse_turing,
            "anti-thrall": ChallengeAdapter.generate_anti_thrall,
            "agency": ChallengeAdapter.generate_agency,
            "counter-coaching": ChallengeAdapter.generate_counter_coaching,
            "intent-provenance": ChallengeAdapter.generate_intent_provenance,
        }

        # Track async suites that need await
        llm_dynamic_pending = False

        for suite in resolved_suites:
            if suite == LLM_DYNAMIC_SUITE:
                llm_dynamic_pending = True
                continue
            elif suite == MULTI_ROUND_SUITE:
                client, server = ChallengeAdapter.generate_novel_reasoning(difficulty)
            elif suite == GOVERNANCE_SUITE:
                client, server = ChallengeAdapter.generate_governance()
            elif suite == "intent-provenance" and vcp_token is not None:
                client, server = ChallengeAdapter.generate_intent_provenance(
                    vcp_token=vcp_token
                )
            else:
                gen = generators.get(suite)
                if gen is None:
                    raise ValueError(f"Unknown suite: {suite}")
                client, server = gen()
            client_challenges[suite] = client
            server_answers[suite] = server

        # Generate LLM-dynamic challenges (async -- requires await)
        if llm_dynamic_pending:
            from mettle.llm_challenges import generate_llm_challenges

            client, server = await generate_llm_challenges()
            client_challenges[LLM_DYNAMIC_SUITE] = client
            server_answers[LLM_DYNAMIC_SUITE] = server

        presence_state = None
        issued_client_challenges = client_challenges
        if presence_registration is not None:
            presence_state = new_session_presence(
                session_id=session_id,
                public_key_pem=presence_registration["public_key_pem"],
                audience=presence_registration["audience"],
            )
            first_suite = resolved_suites[0]
            presence_state["client_challenges"] = client_challenges
            presence_state["current_action"] = (
                "round:1"
                if first_suite == MULTI_ROUND_SUITE
                else f"suite:{first_suite}"
            )
            issued_client_challenges = {
                first_suite: attach_continuity_challenge(
                    presence_state,
                    presence_state["current_action"],
                    client_challenges[first_suite],
                )
            }
            # Fail before persistence if the issuer cannot authenticate the
            # initial holder state.
            issuer_signed_session_presence(
                presence_state,
                session_id=session_id,
            )

        # Respondent time begins only once every challenge to be returned is
        # ready. In particular, upstream LLM generation is never charged to the
        # respondent's advertised budget.
        issued_at = time.time()
        now = datetime.fromtimestamp(issued_at, tz=timezone.utc)
        expires_at = datetime.fromtimestamp(
            issued_at + (time_budget_ms / 1000), tz=timezone.utc
        )
        novel_is_issued = MULTI_ROUND_SUITE in resolved_suites and (
            presence_state is None or resolved_suites[0] == MULTI_ROUND_SUITE
        )

        # Store session metadata
        session_meta = {
            "session_id": session_id,
            "user_id": user_id,
            "entity_id": entity_id,
            "vcp_token": vcp_token,
            "operator_commitment": operator_commitment,
            "suites": resolved_suites,
            "difficulty": difficulty,
            "status": SessionStatus.CHALLENGES_GENERATED.value,
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "time_budget_ms": time_budget_ms,
            # The wall-clock budget begins when the server issues challenges,
            # not when the caller chooses to submit its first answer.
            "start_time": issued_at,
            # Each novel-reasoning response is timed from the previous round
            # boundary. Using cumulative session age here would force later
            # rounds to look slower and make an AI acceleration signature
            # impossible through the real API.
            "novel_started_at": issued_at if novel_is_issued else None,
            "round_started_at": issued_at if novel_is_issued else None,
            "current_round": 0,
            "suites_completed": [],
            "suite_results": {},
            "round_data": [],
            "presence": presence_state,
        }

        # Store in Redis
        pipe = self.redis.pipeline()
        pipe.setex(_key(session_id), storage_ttl, json.dumps(session_meta))
        pipe.setex(_key(session_id, "answers"), storage_ttl, json.dumps(server_answers))
        await pipe.execute()
        # The metadata key now proves that this member is live, so the short
        # construction reservation is no longer needed for stale-set pruning.
        delete_fn = getattr(self.redis, "delete", None)
        if callable(delete_fn):
            await delete_fn(_key(session_id, "reservation"))

        return session_id, issued_client_challenges, session_meta

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get session metadata."""
        raw = await self.redis.get(_key(session_id))
        if raw is None:
            return None
        session = self._decode_redis_object(raw, "session metadata")
        self._validate_session_envelope(session, session_id)
        return session

    async def get_session_answers(self, session_id: str) -> dict[str, Any] | None:
        """Get server-side answers for a session."""
        raw = await self.redis.get(_key(session_id, "answers"))
        if raw is None:
            return None
        return self._decode_redis_object(raw, "session answers")

    async def cancel_session(self, session_id: str, user_id: str) -> bool:
        """Cancel a session. Returns False if not found or wrong user."""
        async with self._session_lock(session_id):
            return await self._cancel_session(session_id, user_id)

    async def _cancel_session(self, session_id: str, user_id: str) -> bool:
        session = await self.get_session(session_id)
        if session is None:
            return False
        if session["user_id"] != user_id:
            return False
        if session["status"] in (
            SessionStatus.COMPLETED.value,
            SessionStatus.CANCELLED.value,
        ):
            return False

        session["status"] = SessionStatus.CANCELLED.value
        session["terminated_at"] = time.time()
        await self._persist_terminal_session(session)
        return True

    # ---- Single-Shot Verification ----

    async def verify_single_shot(
        self,
        session_id: str,
        suite: str,
        answers: dict[str, Any],
        presence_proof: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Verify answers for a single-shot suite.

        Returns evaluation results. Raises ValueError on invalid state.
        """
        async with self._session_lock(session_id):
            return await self._verify_single_shot(
                session_id, suite, answers, presence_proof
            )

    async def _verify_single_shot(
        self,
        session_id: str,
        suite: str,
        answers: dict[str, Any],
        presence_proof: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = await self.get_session(session_id)
        if session is None:
            raise ValueError("Session not found")

        if session["status"] not in (
            SessionStatus.CHALLENGES_GENERATED.value,
            SessionStatus.IN_PROGRESS.value,
        ):
            raise ValueError(f"Session not in verifiable state: {session['status']}")

        await self._enforce_time_budget(session)

        if suite not in session["suites"]:
            raise ValueError(f"Suite '{suite}' not in this session")

        if suite in session["suites_completed"]:
            raise ValueError(f"Suite '{suite}' already completed")

        if suite == MULTI_ROUND_SUITE:
            raise ValueError("Novel reasoning requires multi-round endpoint")

        session["status"] = SessionStatus.IN_PROGRESS.value
        presence_message = verify_submission_proof(
            presence=session.get("presence"),
            proof=presence_proof,
            session_id=session_id,
            action=f"suite:{suite}",
            answers=answers,
        )
        verify_continuity_answer(session.get("presence"), f"suite:{suite}", answers)
        evaluation_answers = dict(answers)
        evaluation_answers.pop(CONTINUITY_ANSWER_KEY, None)

        # Get server answers and evaluate
        server_answers = await self.get_session_answers(session_id)
        if server_answers is None:
            raise ValueError("Session answers expired")
        if set(server_answers) != set(session["suites"]):
            raise SessionStateError(
                "Corrupt session answers: suite set does not match session"
            )
        suite_server = server_answers.get(suite)
        if not isinstance(suite_server, dict):
            raise SessionStateError(f"Corrupt session answers for suite {suite}")

        # LLM-dynamic suite requires async evaluation
        if suite == LLM_DYNAMIC_SUITE:
            from mettle.llm_challenges import evaluate_llm_challenges

            elapsed_ms = max(
                0,
                int((time.time() - float(session["start_time"])) * 1000),
            )
            result = await evaluate_llm_challenges(
                evaluation_answers, suite_server, response_time_ms=elapsed_ms
            )
        else:
            result = ChallengeAdapter.evaluate_single_shot(
                suite, evaluation_answers, suite_server
            )

        # Update session
        session["suites_completed"].append(suite)
        session["suite_results"][suite] = result

        # Check if all suites completed
        if set(session["suites_completed"]) == set(session["suites"]):
            session["status"] = SessionStatus.COMPLETED.value
            session["completed_at"] = time.time()
            ttl = COMPLETED_SESSION_TTL
        else:
            ttl = self._remaining_active_ttl(session)

        if presence_message is not None:
            if presence_proof is None:
                raise RuntimeError("Verified presence proof unexpectedly missing")
            advance_session_presence(
                presence=session["presence"],
                message=presence_message,
                signature=presence_proof["signature"],
                action=f"suite:{suite}",
            )

        next_challenge = (
            self._advance_presence_suite(session) if session.get("presence") else None
        )

        response_presence = issuer_signed_session_presence(
            session.get("presence"),
            session_id=session_id,
            completed=session["status"] == SessionStatus.COMPLETED.value,
        )
        if session["status"] == SessionStatus.COMPLETED.value:
            await self._persist_terminal_session(session)
        else:
            await self._persist_active_session(session, ttl)
        response = dict(result)
        response["presence"] = response_presence
        response["next_challenge"] = next_challenge
        return response

    # ---- Multi-Round (Suite 10) ----

    async def submit_round_answer(
        self,
        session_id: str,
        round_num: int,
        answers: dict[str, Any],
        presence_proof: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit answers for a multi-round challenge round.

        Returns round evaluation with feedback. Raises ValueError on invalid state.
        """
        async with self._session_lock(session_id):
            return await self._submit_round_answer(
                session_id, round_num, answers, presence_proof
            )

    async def _submit_round_answer(
        self,
        session_id: str,
        round_num: int,
        answers: dict[str, Any],
        presence_proof: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = await self.get_session(session_id)
        if session is None:
            raise ValueError("Session not found")

        if MULTI_ROUND_SUITE not in session["suites"]:
            raise ValueError("Session does not include novel-reasoning suite")

        if session["status"] not in (
            SessionStatus.CHALLENGES_GENERATED.value,
            SessionStatus.IN_PROGRESS.value,
        ):
            raise ValueError(f"Session not in answerable state: {session['status']}")

        await self._enforce_time_budget(session)

        expected_round = session["current_round"] + 1
        if round_num != expected_round:
            raise ValueError(f"Expected round {expected_round}, got {round_num}")

        session["status"] = SessionStatus.IN_PROGRESS.value
        presence_message = verify_submission_proof(
            presence=session.get("presence"),
            proof=presence_proof,
            session_id=session_id,
            action=f"round:{round_num}",
            answers=answers,
        )
        verify_continuity_answer(session.get("presence"), f"round:{round_num}", answers)
        evaluation_answers = dict(answers)
        evaluation_answers.pop(CONTINUITY_ANSWER_KEY, None)

        # The session-wide budget was enforced above. Novel reasoning has its
        # own clock beginning when that suite is actually issued, while curve
        # analysis records each individual round's duration.
        answered_at = time.time()
        novel_started_at = float(
            session.get("novel_started_at") or session["start_time"]
        )
        elapsed_ms = (answered_at - novel_started_at) * 1000
        round_started_at = float(session.get("round_started_at") or novel_started_at)
        round_response_ms = max(0.0, (answered_at - round_started_at) * 1000)

        server_answers = await self.get_session_answers(session_id)
        if server_answers is None:
            raise ValueError("Session answers expired")
        if set(server_answers) != set(session["suites"]):
            raise SessionStateError(
                "Corrupt session answers: suite set does not match session"
            )
        novel_server = server_answers.get(MULTI_ROUND_SUITE)
        if not isinstance(novel_server, dict):
            raise SessionStateError("Corrupt novel-reasoning answers")
        raw_time_budget = novel_server.get("time_budget_s")
        num_rounds = novel_server.get("num_rounds")
        novel_challenges = novel_server.get("challenges")
        if (
            isinstance(raw_time_budget, bool)
            or not isinstance(raw_time_budget, int | float)
            or not math.isfinite(float(raw_time_budget))
            or float(raw_time_budget) <= 0
            or isinstance(num_rounds, bool)
            or not isinstance(num_rounds, int)
            or num_rounds not in {2, 3}
            or not isinstance(novel_challenges, dict)
            or not novel_challenges
            or any(
                not isinstance(name, str) or not name or not isinstance(challenge, dict)
                for name, challenge in novel_challenges.items()
            )
        ):
            raise SessionStateError("Corrupt novel-reasoning answers")
        time_budget_ms = float(raw_time_budget) * 1000

        time_exceeded = elapsed_ms >= time_budget_ms
        time_remaining_ms = max(0, int(time_budget_ms - elapsed_ms))

        # Evaluate each challenge for this round
        round_results: dict[str, Any] = {}
        all_errors: list[str] = []
        total_accuracy = 0.0
        num_challenges = 0

        challenge_answers = evaluation_answers.get("challenges", evaluation_answers)
        if not isinstance(challenge_answers, dict) or any(
            not isinstance(name, str)
            or not name
            or not isinstance(challenge_answer, dict)
            for name, challenge_answer in challenge_answers.items()
        ):
            raise ValueError(
                "Round answers must be an object mapping challenge names to objects"
            )
        expected_challenges = set(novel_challenges)
        if set(challenge_answers) != expected_challenges:
            missing = sorted(expected_challenges - set(challenge_answers))
            unexpected = sorted(set(challenge_answers) - expected_challenges)
            raise ValueError(
                "Round answers must cover the complete issued challenge set; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for challenge_name, challenge_answers_data in challenge_answers.items():
            result = ChallengeAdapter.evaluate_novel_round(
                challenge_name, round_num, challenge_answers_data, novel_server
            )
            round_results[challenge_name] = result
            all_errors.extend(result.get("errors", []))
            total_accuracy += result.get("accuracy", 0.0)
            num_challenges += 1

        avg_accuracy = total_accuracy / num_challenges if num_challenges > 0 else 0.0

        # Record round data
        round_record = {
            "round": round_num,
            "response_time_ms": round(round_response_ms, 1),
            "accuracy": round(avg_accuracy, 4),
            "time_exceeded": time_exceeded,
            "results": round_results,
        }
        session["round_data"].append(round_record)
        session["current_round"] = round_num
        session["round_started_at"] = answered_at

        # Build feedback
        is_final_round = round_num >= num_rounds
        feedback: dict[str, Any] = {
            "accuracy": round(avg_accuracy, 4),
            "challenge_feedback": round_results,
        }

        # Determine next round data
        next_round_data = None
        if not is_final_round:
            next_round_data = self._next_novel_round_data(round_num + 1, novel_server)

        if is_final_round:
            # Analyze iteration curve
            curve_result = self._analyze_iteration_curve(
                session["round_data"], novel_server
            )
            if any(rd.get("time_exceeded", False) for rd in session["round_data"]):
                curve_result["passed"] = False
                curve_result["details"]["time_exceeded"] = True
            session["suite_results"][MULTI_ROUND_SUITE] = curve_result
            session["suites_completed"].append(MULTI_ROUND_SUITE)

            # Check if all suites completed
            if set(session["suites_completed"]) == set(session["suites"]):
                session["status"] = SessionStatus.COMPLETED.value
                session["completed_at"] = time.time()

        ttl = (
            COMPLETED_SESSION_TTL
            if session["status"] == SessionStatus.COMPLETED.value
            else self._remaining_active_ttl(session)
        )
        if presence_message is not None:
            if presence_proof is None:
                raise RuntimeError("Verified presence proof unexpectedly missing")
            advance_session_presence(
                presence=session["presence"],
                message=presence_message,
                signature=presence_proof["signature"],
                action=f"round:{round_num}",
            )
        next_challenge = None
        if session.get("presence"):
            if is_final_round:
                next_challenge = self._advance_presence_suite(session)
            else:
                session["presence"]["current_action"] = f"round:{round_num + 1}"
                if next_round_data is None:
                    raise RuntimeError("Next novel-reasoning round data is missing")
                next_round_data = attach_continuity_challenge(
                    session["presence"],
                    session["presence"]["current_action"],
                    next_round_data,
                )
        response_presence = issuer_signed_session_presence(
            session.get("presence"),
            session_id=session_id,
            completed=session["status"] == SessionStatus.COMPLETED.value,
        )
        if session["status"] == SessionStatus.COMPLETED.value:
            await self._persist_terminal_session(session)
        else:
            await self._persist_active_session(session, ttl)

        return {
            "round_num": round_num,
            "accuracy": round(avg_accuracy, 4),
            "errors": all_errors[:10],
            "feedback": feedback,
            "time_remaining_ms": time_remaining_ms,
            "next_round_data": next_round_data,
            "presence": response_presence,
            "next_challenge": next_challenge,
        }

    async def get_round_feedback(
        self, session_id: str, round_num: int
    ) -> dict[str, Any] | None:
        """Get feedback for a completed round."""
        session = await self.get_session(session_id)
        if session is None:
            return None

        for rd in session.get("round_data", []):
            if rd["round"] == round_num:
                return rd
        return None

    # ---- Results ----

    async def get_result(self, session_id: str) -> dict[str, Any] | None:
        """Get final results for a session. Returns None if not completed."""
        session = await self.get_session(session_id)
        if session is None:
            return None

        if session["status"] != SessionStatus.COMPLETED.value:
            return None

        results, start_time, completed_at = self._validate_completed_session(
            session, session_id
        )
        elapsed_ms = max(0, int((completed_at - start_time) * 1000))

        # Check overall pass
        all_passed = all(result["passed"] for result in results.values())

        # Extract iteration curve if novel reasoning was run
        iteration_curve = None
        novel_result = results.get(MULTI_ROUND_SUITE)
        if novel_result:
            iteration_curve = novel_result.get("iteration_curve")

        return {
            "session_id": session_id,
            "status": session["status"],
            "suites_completed": session["suites_completed"],
            "results": results,
            "overall_passed": all_passed,
            "iteration_curve": iteration_curve,
            "elapsed_ms": elapsed_ms,
            "presence": issuer_signed_session_presence(
                session.get("presence"), session_id=session_id, completed=True
            ),
        }

    # ---- Credential Presentation ----

    @staticmethod
    def _advance_presence_suite(session: dict[str, Any]) -> dict[str, Any] | None:
        """Issue only the next suite challenge in a Presence session."""
        presence = session["presence"]
        if session["status"] == SessionStatus.COMPLETED.value:
            presence["current_action"] = None
            retire_continuity_secret(presence)
            return None
        completed = set(session["suites_completed"])
        for suite in session["suites"]:
            if suite in completed:
                continue
            presence["current_action"] = (
                f"round:{session.get('current_round', 0) + 1}"
                if suite == MULTI_ROUND_SUITE
                else f"suite:{suite}"
            )
            if suite == MULTI_ROUND_SUITE and session.get("novel_started_at") is None:
                issued_at = time.time()
                session["novel_started_at"] = issued_at
                session["round_started_at"] = issued_at
            return {
                suite: attach_continuity_challenge(
                    presence,
                    presence["current_action"],
                    presence["client_challenges"][suite],
                )
            }
        raise RuntimeError("Presence session has no challenge left to issue")

    async def create_presentation_challenge(
        self,
        *,
        verifier_user_id: str,
        credential_jti: str,
        audience: str,
    ) -> dict[str, Any]:
        """Persist a fresh verifier-owned proof-of-possession challenge."""
        rate_key = _presentation_rate_key(verifier_user_id)
        eval_fn = getattr(self.redis, "eval", None)
        now_ms = int(time.time() * 1000)
        if callable(eval_fn):
            rate_result = await eval_fn(
                _PRESENTATION_RATE_SCRIPT,
                1,
                rate_key,
                now_ms,
                PRESENTATION_RATE_WINDOW_MS,
                MAX_PRESENTATION_CHALLENGES_PER_MINUTE,
                secrets.token_urlsafe(18),
            )
            if not isinstance(rate_result, (list, tuple)) or len(rate_result) != 2:
                raise RedisError("Invalid presentation rate-limit result")
            count, retry_ms = (int(rate_result[0]), int(rate_result[1]))
        else:
            count, retry_ms = await self._reserve_presentation_rate_fallback(
                rate_key, now_ms
            )
        if count < 0:
            raise SessionRateLimitError(
                "Presentation challenge rate limit exceeded",
                max(1, math.ceil(retry_ms / 1000)),
            )

        challenge_id = secrets.token_urlsafe(32)
        expires_at = datetime.fromtimestamp(
            time.time() + PRESENTATION_CHALLENGE_TTL, tz=timezone.utc
        ).isoformat()
        challenge = {
            "challenge_id": challenge_id,
            "nonce": secrets.token_urlsafe(32),
            "audience": audience,
            "credential_jti": credential_jti,
            "verifier_user_id": verifier_user_id,
            "expires_at": expires_at,
        }
        await self.redis.setex(
            _presentation_key(challenge_id),
            PRESENTATION_CHALLENGE_TTL,
            json.dumps(challenge),
        )
        return challenge

    async def verify_presentation(
        self,
        *,
        verifier_user_id: str,
        challenge_id: str,
        credential_jti: str,
        audience: str,
        public_key_pem: str,
        holder_signature: str,
    ) -> dict[str, Any]:
        """Verify and atomically consume one live presentation challenge."""
        async with self._session_lock(f"presentation:{challenge_id}"):
            raw = await self.redis.get(_presentation_key(challenge_id))
            if raw is None:
                raise ValueError("Presentation challenge expired or already used")
            challenge = self._decode_redis_object(raw, "presentation challenge")
            if challenge.get("verifier_user_id") != verifier_user_id:
                raise ValueError("Presentation challenge belongs to another verifier")
            if challenge.get("credential_jti") != credential_jti:
                raise ValueError("Credential does not match presentation challenge")
            if challenge.get("audience") != audience:
                raise ValueError(
                    "Credential audience does not match presentation challenge"
                )
            verify_holder_signature(
                public_key_pem=public_key_pem,
                signature=holder_signature,
                challenge=challenge,
            )
            eval_fn = getattr(self.redis, "eval", None)
            if callable(eval_fn):
                deleted = await eval_fn(
                    _PRESENTATION_CONSUME_SCRIPT,
                    1,
                    _presentation_key(challenge_id),
                    raw,
                )
            else:
                delete = getattr(self.redis, "delete", None)
                if not callable(delete):
                    raise RuntimeError(
                        "Redis client cannot consume presentation challenges"
                    )
                deleted = await delete(_presentation_key(challenge_id))
            if deleted != 1:
                raise ValueError("Presentation challenge expired or already used")
            return challenge

    # ---- Iteration Curve Analysis ----

    def _analyze_iteration_curve(
        self,
        round_data: list[dict[str, Any]],
        server_answers: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze the iteration improvement curve for substrate detection."""
        from scripts.engine import IterationCurveAnalyzer

        # Format round data for the analyzer
        analyzer_rounds = []
        for rd in round_data:
            analyzer_rounds.append(
                {
                    "round": rd["round"],
                    "response_time_ms": rd["response_time_ms"],
                    "accuracy": rd["accuracy"],
                    "structural_change": abs(
                        rd["accuracy"]
                        - (
                            round_data[round_data.index(rd) - 1]["accuracy"]
                            if round_data.index(rd) > 0
                            else 0
                        )
                    ),
                    "error_magnitude": 1.0 - rd["accuracy"],
                }
            )

        curve = IterationCurveAnalyzer.analyze_curve(analyzer_rounds)
        pass_threshold = server_answers.get("pass_threshold", 0.65)
        passed = curve["overall"] > pass_threshold and curve["signature"] == "AI"

        return {
            "passed": passed,
            "score": curve["overall"],
            "iteration_curve": curve,
            "round_data": round_data,
            "details": {
                "signature": curve["signature"],
                "threshold": pass_threshold,
            },
        }

    # ---- Rate Limiting ----

    async def _check_rate_limits(self, user_id: str) -> None:
        """Check rate limits for session creation. Raises ValueError if exceeded."""
        await self._prune_stale_active_sessions(user_id)
        # Check active sessions
        active_count = await self.redis.scard(_rate_key(user_id, "active"))
        if active_count is not None and active_count >= MAX_ACTIVE_SESSIONS_PER_USER:
            raise SessionRateLimitError(
                f"Maximum active sessions ({MAX_ACTIVE_SESSIONS_PER_USER}) exceeded. "
                "Complete or cancel existing sessions.",
                ACTIVE_SESSION_TTL,
            )

        # Check hourly limit
        hourly_raw = await self.redis.get(_rate_key(user_id, "hourly"))
        hourly_count = int(hourly_raw) if hourly_raw else 0
        if hourly_count >= MAX_SESSIONS_PER_HOUR:
            raise SessionRateLimitError(
                f"Hourly session limit ({MAX_SESSIONS_PER_HOUR}) exceeded. Try again later.",
                RATE_LIMIT_WINDOW,
            )

    async def _reserve_rate_limits(
        self,
        user_id: str,
        session_id: str,
        *,
        active_ttl: int = ACTIVE_SESSION_TTL,
    ) -> None:
        """Atomically reserve active-session and hourly quota in Redis."""
        eval_fn = getattr(self.redis, "eval", None)
        if not callable(eval_fn):
            # Lightweight repository fakes do not implement Lua. Production
            # redis.asyncio clients always do. Keep the fallback behavior
            # semantically equivalent for unit tests.
            async with self._local_lock(f"rate:{user_id}"):
                await self._check_rate_limits(user_id)
                await self.redis.setex(_key(session_id, "reservation"), active_ttl, "1")
                pipe = self.redis.pipeline()
                pipe.sadd(_rate_key(user_id, "active"), session_id)
                pipe.expire(_rate_key(user_id, "active"), active_ttl)
                pipe.incr(_rate_key(user_id, "hourly"))
                pipe.expire(_rate_key(user_id, "hourly"), RATE_LIMIT_WINDOW)
                await pipe.execute()
            return

        result = await eval_fn(
            _RATE_RESERVATION_SCRIPT,
            3,
            _rate_key(user_id, "active"),
            _rate_key(user_id, "hourly"),
            _key(session_id, "reservation"),
            session_id,
            MAX_ACTIVE_SESSIONS_PER_USER,
            MAX_SESSIONS_PER_HOUR,
            active_ttl,
            RATE_LIMIT_WINDOW,
            f"{settings.redis_namespace}:session:",
        )
        if result == -1:
            raise SessionRateLimitError(
                f"Maximum active sessions ({MAX_ACTIVE_SESSIONS_PER_USER}) exceeded. "
                "Complete or cancel existing sessions.",
                ACTIVE_SESSION_TTL,
            )
        if result == -2:
            raise SessionRateLimitError(
                f"Hourly session limit ({MAX_SESSIONS_PER_HOUR}) exceeded. Try again later.",
                RATE_LIMIT_WINDOW,
            )

    async def _release_rate_reservation(self, user_id: str, session_id: str) -> None:
        """Release active and hourly quota after failed session construction."""
        eval_fn = getattr(self.redis, "eval", None)
        if callable(eval_fn):
            await eval_fn(
                _RATE_RESERVATION_RELEASE_SCRIPT,
                3,
                _rate_key(user_id, "active"),
                _rate_key(user_id, "hourly"),
                _key(session_id, "reservation"),
                session_id,
            )
            return

        await self.redis.srem(_rate_key(user_id, "active"), session_id)
        delete_fn = getattr(self.redis, "delete", None)
        if callable(delete_fn):
            await delete_fn(_key(session_id, "reservation"))
        decr_fn = getattr(self.redis, "decr", None)
        hourly_raw = await self.redis.get(_rate_key(user_id, "hourly"))
        if callable(decr_fn) and hourly_raw and int(hourly_raw) > 0:
            await decr_fn(_rate_key(user_id, "hourly"))

    @asynccontextmanager
    async def _session_lock(self, session_id: str):
        """Serialize transitions with a renewable, token-owned Redis lease."""
        set_fn = getattr(self.redis, "set", None)
        eval_fn = getattr(self.redis, "eval", None)
        if not callable(set_fn) or not callable(eval_fn):
            # Repository fakes share this lock through the Redis object. This
            # preserves the serialization contract without pretending to be a
            # distributed lease.
            async with self._local_lock(f"session:{session_id}"):
                lease = _SessionLease(_key(session_id, "lock"), None)
                context_token = _CURRENT_SESSION_LEASE.set(lease)
                try:
                    yield lease
                finally:
                    _CURRENT_SESSION_LEASE.reset(context_token)
            return

        lock_key = _key(session_id, "lock")
        token = secrets.token_urlsafe(24)
        acquired = await set_fn(lock_key, token, nx=True, ex=SESSION_LOCK_TTL)
        if not acquired:
            raise ValueError("Session operation already in progress")
        lease = _SessionLease(lock_key, token)
        renewal = asyncio.create_task(self._renew_session_lock(lease))
        context_token = _CURRENT_SESSION_LEASE.set(lease)
        try:
            yield lease
            await self._assert_lease_owned(lease)
        finally:
            _CURRENT_SESSION_LEASE.reset(context_token)
            renewal.cancel()
            try:
                await renewal
            except asyncio.CancelledError:
                pass
            try:
                await eval_fn(_LOCK_RELEASE_SCRIPT, 1, lock_key, token)
            except RedisError:
                logger.warning("Unable to release session lock %s", lock_key)

    async def _renew_session_lock(self, lease: _SessionLease) -> None:
        """Renew a lease until its owner exits or ownership is lost."""
        if lease.token is None:
            return
        try:
            while True:
                await asyncio.sleep(SESSION_LOCK_RENEW_INTERVAL)
                renewed = await self.redis.eval(
                    _LOCK_RENEW_SCRIPT,
                    1,
                    lease.lock_key,
                    lease.token,
                    SESSION_LOCK_TTL,
                )
                if int(renewed) != 1:
                    lease.lost.set()
                    return
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            lease.renewal_error = exc
            lease.lost.set()

    async def _assert_lease_owned(self, lease: _SessionLease | None = None) -> None:
        """Fail closed unless the current worker still owns its lease."""
        current = lease or _CURRENT_SESSION_LEASE.get()
        if current is None:
            raise SessionLockLostError("Session state commit attempted without a lock")
        if current.lost.is_set():
            detail = " after renewal failure" if current.renewal_error else ""
            raise SessionLockLostError(f"Session lock ownership was lost{detail}")
        if current.token is None:
            return
        owned = await self.redis.eval(
            _LOCK_CHECK_SCRIPT, 1, current.lock_key, current.token
        )
        if int(owned) != 1:
            current.lost.set()
            raise SessionLockLostError("Session lock ownership was lost")

    @asynccontextmanager
    async def _local_lock(self, name: str):
        """Share a bounded in-process lock among managers using one fake client."""
        lock = self._local_locks.setdefault(name, asyncio.Lock())
        async with lock:
            yield

    # ---- Helpers ----

    @staticmethod
    def _session_time_budget_ms(suites: list[str], difficulty: str) -> int:
        """Return the exact respondent budget without counting novel twice."""
        from scripts.engine import NovelReasoningChallenges

        params = NovelReasoningChallenges.DIFFICULTY_PARAMS.get(difficulty)
        if params is None:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        single_shot_count = len(suites) - int(MULTI_ROUND_SUITE in suites)
        budget_ms = single_shot_count * 30_000
        if MULTI_ROUND_SUITE in suites:
            budget_ms += int(params["time_budget_s"]) * 1000
        if budget_ms <= 0:
            raise ValueError("At least one suite is required")
        return budget_ms

    @staticmethod
    def _decode_redis_object(raw: Any, label: str) -> dict[str, Any]:
        """Decode an untrusted Redis value as a JSON object or fail closed."""
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as exc:
            raise SessionStateError(f"Corrupt {label}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise SessionStateError(f"Corrupt {label}: expected a JSON object")
        return value

    @staticmethod
    def _validate_session_envelope(
        session: dict[str, Any], expected_session_id: str
    ) -> None:
        """Validate lifecycle fields required by every session operation."""
        if session.get("session_id") != expected_session_id:
            raise SessionStateError("Corrupt session metadata: session_id mismatch")
        if not isinstance(session.get("user_id"), str) or not session["user_id"]:
            raise SessionStateError("Corrupt session metadata: invalid user_id")
        suites = session.get("suites")
        if (
            not isinstance(suites, list)
            or not suites
            or any(not isinstance(suite, str) for suite in suites)
            or len(suites) != len(set(suites))
            or any(suite not in SUITE_NAMES for suite in suites)
        ):
            raise SessionStateError("Corrupt session metadata: invalid suites")
        valid_statuses = {status.value for status in SessionStatus}
        if session.get("status") not in valid_statuses:
            raise SessionStateError("Corrupt session metadata: invalid status")
        completed = session.get("suites_completed")
        if (
            not isinstance(completed, list)
            or any(not isinstance(suite, str) for suite in completed)
            or len(completed) != len(set(completed))
            or not set(completed).issubset(set(suites))
        ):
            raise SessionStateError(
                "Corrupt session metadata: invalid suites_completed"
            )
        if not isinstance(session.get("suite_results"), dict):
            raise SessionStateError("Corrupt session metadata: invalid suite_results")
        if not isinstance(session.get("round_data"), list):
            raise SessionStateError("Corrupt session metadata: invalid round_data")
        current_round = session.get("current_round")
        if (
            isinstance(current_round, bool)
            or not isinstance(current_round, int)
            or current_round < 0
        ):
            raise SessionStateError("Corrupt session metadata: invalid current_round")

    @staticmethod
    def _finite_timestamp(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise SessionStateError(f"Corrupt completed session: invalid {label}")
        timestamp = float(value)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise SessionStateError(f"Corrupt completed session: invalid {label}")
        return timestamp

    @classmethod
    def _validate_completed_session(
        cls, session: dict[str, Any], expected_session_id: str
    ) -> tuple[dict[str, dict[str, Any]], float, float]:
        """Validate a completed result record before deriving credentials."""
        cls._validate_session_envelope(session, expected_session_id)
        if session["status"] != SessionStatus.COMPLETED.value:
            raise SessionStateError("Session is not completed")
        suites = session["suites"]
        if set(session["suites_completed"]) != set(suites):
            raise SessionStateError(
                "Corrupt completed session: suites are not fully completed"
            )
        results = session["suite_results"]
        if set(results) != set(suites):
            raise SessionStateError(
                "Corrupt completed session: result set does not match suites"
            )
        for suite, result in results.items():
            if not isinstance(result, dict) or not isinstance(
                result.get("passed"), bool
            ):
                raise SessionStateError(
                    f"Corrupt completed session: invalid result for {suite}"
                )
        start_time = cls._finite_timestamp(session.get("start_time"), "start_time")
        completed_at = cls._finite_timestamp(
            session.get("completed_at"), "completed_at"
        )
        if completed_at < start_time:
            raise SessionStateError(
                "Corrupt completed session: completed_at predates start_time"
            )
        return results, start_time, completed_at

    async def _persist_active_session(self, session: dict[str, Any], ttl: int) -> None:
        """Persist a nonterminal transition only while its lock is owned."""
        lease = _CURRENT_SESSION_LEASE.get()
        await self._assert_lease_owned(lease)
        payload = json.dumps(session)
        if lease is not None and lease.token is not None:
            result = await self.redis.eval(
                _ACTIVE_PERSIST_SCRIPT,
                2,
                _key(session["session_id"]),
                lease.lock_key,
                payload,
                ttl,
                lease.token,
            )
            if int(result) != 1:
                lease.lost.set()
                raise SessionLockLostError("Session lock ownership was lost")
            return
        await self.redis.setex(_key(session["session_id"]), ttl, payload)

    async def _persist_terminal_session(self, session: dict[str, Any]) -> None:
        """Atomically persist terminal state and remove all active secrets."""
        status = session.get("status")
        if status not in {
            SessionStatus.COMPLETED.value,
            SessionStatus.EXPIRED.value,
            SessionStatus.CANCELLED.value,
        }:
            raise SessionStateError("Refusing to persist a nonterminal session")
        session_id = session.get("session_id")
        user_id = session.get("user_id")
        if not isinstance(session_id, str) or not isinstance(user_id, str):
            raise SessionStateError("Corrupt terminal session identity")
        lease = _CURRENT_SESSION_LEASE.get()
        await self._assert_lease_owned(lease)
        payload = json.dumps(session)
        if lease is not None and lease.token is not None:
            result = await self.redis.eval(
                _TERMINAL_PERSIST_SCRIPT,
                5,
                _key(session_id),
                _rate_key(user_id, "active"),
                _key(session_id, "answers"),
                lease.lock_key,
                _key(session_id, "reservation"),
                payload,
                COMPLETED_SESSION_TTL,
                session_id,
                lease.token,
            )
            if int(result) != 1:
                if int(result) == -2:
                    raise SessionStateError(
                        "Corrupt active-session index prevents terminal commit"
                    )
                lease.lost.set()
                raise SessionLockLostError("Session lock ownership was lost")
            return

        # Lightweight fakes have no Lua. Their pipeline is the explicit atomic
        # boundary for the same three state changes.
        try:
            pipe = self.redis.pipeline(transaction=True)
        except TypeError:
            pipe = self.redis.pipeline()
        pipe.setex(_key(session_id), COMPLETED_SESSION_TTL, payload)
        pipe.srem(_rate_key(user_id, "active"), session_id)
        delete_in_pipeline = callable(getattr(pipe, "delete", None))
        if delete_in_pipeline:
            pipe.delete(_key(session_id, "answers"))
            pipe.delete(_key(session_id, "reservation"))
        await pipe.execute()
        if not delete_in_pipeline:
            delete_fn = getattr(self.redis, "delete", None)
            if not callable(delete_fn):
                raise RedisError("Redis client cannot delete terminal session secrets")
            await delete_fn(_key(session_id, "answers"))
            await delete_fn(_key(session_id, "reservation"))

    async def _prune_stale_active_sessions(self, user_id: str) -> None:
        """Remove active-set members with neither metadata nor a reservation."""
        smembers = getattr(self.redis, "smembers", None)
        if not callable(smembers):
            return
        active_key = _rate_key(user_id, "active")
        members = await smembers(active_key)
        for raw_member in members or ():
            member = (
                raw_member.decode("utf-8")
                if isinstance(raw_member, bytes)
                else str(raw_member)
            )
            if await self.redis.get(_key(member)) is not None:
                continue
            if await self.redis.get(_key(member, "reservation")) is not None:
                continue
            await self.redis.srem(active_key, member)

    async def _reserve_presentation_rate_fallback(
        self, rate_key: str, now_ms: int
    ) -> tuple[int, int]:
        """Sliding-window limiter for bounded in-memory Redis test doubles."""
        async with self._local_lock(f"presentation-rate:{rate_key}"):
            raw = await self.redis.get(rate_key)
            if raw is None:
                timestamps: list[int] = []
            else:
                value = self._decode_redis_object(raw, "presentation rate state")
                candidate = value.get("timestamps")
                if not isinstance(candidate, list) or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in candidate
                ):
                    raise SessionStateError("Corrupt presentation rate state")
                timestamps = candidate
            cutoff = now_ms - PRESENTATION_RATE_WINDOW_MS
            timestamps = [stamp for stamp in timestamps if stamp > cutoff]
            if len(timestamps) >= MAX_PRESENTATION_CHALLENGES_PER_MINUTE:
                retry_ms = max(1, timestamps[0] + PRESENTATION_RATE_WINDOW_MS - now_ms)
                return -1, retry_ms
            timestamps.append(now_ms)
            await self.redis.setex(
                rate_key,
                math.ceil(PRESENTATION_RATE_WINDOW_MS / 1000),
                json.dumps({"timestamps": timestamps}),
            )
            return len(timestamps), 0

    @staticmethod
    def _next_novel_round_data(
        next_round: int, novel_server: dict[str, Any]
    ) -> dict[str, Any]:
        """Project the next round through the adapter's disclosure boundary."""
        builder = getattr(ChallengeAdapter, "build_novel_round_client_data", None)
        if not callable(builder):
            raise SessionStateError("Novel round projection is unavailable")
        try:
            data = builder(next_round, novel_server)
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionStateError("Corrupt novel-reasoning round state") from exc
        if not isinstance(data, dict):
            raise SessionStateError("Novel round builder returned invalid data")
        return data

    @staticmethod
    def _resolve_suites(suites: list[str]) -> list[str]:
        """Resolve 'all' to full suite list and validate names."""
        if len(suites) != len(set(suites)):
            raise ValueError("Duplicate suites are not allowed")
        if "all" in suites and len(suites) != 1:
            raise ValueError("'all' cannot be combined with explicit suites")
        if "all" in suites:
            # Exclude llm-dynamic from "all" when API key isn't available
            resolved = list(SUITE_NAMES)
            if not llm_available():
                resolved = [s for s in resolved if s != LLM_DYNAMIC_SUITE]
            return resolved

        invalid = [s for s in suites if s not in SUITE_NAMES]
        if invalid:
            raise ValueError(f"Unknown suites: {invalid}. Valid: {SUITE_NAMES}")

        return suites

    @staticmethod
    def _remaining_active_ttl(session: dict[str, Any]) -> int:
        """Return a TTL that cannot extend the server-issued absolute expiry."""
        try:
            expires_at = datetime.fromisoformat(session["expires_at"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionStateError(
                "Corrupt session metadata: invalid expires_at"
            ) from exc
        if expires_at.tzinfo is None:
            raise SessionStateError(
                "Corrupt session metadata: expires_at lacks timezone"
            )
        # Redis must retain the session for its entire advertised budget.  A
        # fixed ACTIVE_SESSION_TTL cap silently shortened large, multi-suite
        # sessions after their first state transition.  Ceiling also avoids
        # expiring a key fractionally before the authoritative deadline.
        remaining = math.ceil(expires_at.timestamp() - time.time())
        return max(1, remaining)

    async def _enforce_time_budget(self, session: dict[str, Any]) -> None:
        """Expire sessions whose authoritative wall-clock budget has elapsed."""
        start_time = session.get("start_time")
        budget = session.get("time_budget_ms")
        if (
            isinstance(start_time, bool)
            or not isinstance(start_time, int | float)
            or not math.isfinite(float(start_time))
            or isinstance(budget, bool)
            or not isinstance(budget, int | float)
            or not math.isfinite(float(budget))
            or float(budget) <= 0
        ):
            raise SessionStateError("Corrupt session metadata: invalid time budget")
        elapsed_ms = max(0.0, (time.time() - float(start_time)) * 1000)
        if elapsed_ms < float(budget):
            return

        session["status"] = SessionStatus.EXPIRED.value
        session["terminated_at"] = time.time()
        await self._persist_terminal_session(session)
        raise ValueError("Session time budget exceeded")
