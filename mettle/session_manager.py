"""Session manager for METTLE API.

Handles session lifecycle, Redis storage, timing enforcement, and rate limiting.
Sessions follow the state machine: CREATED -> CHALLENGES_GENERATED -> IN_PROGRESS -> COMPLETED/EXPIRED/CANCELLED
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from datetime import datetime, timezone
from typing import Any

from mettle.challenge_adapter import ChallengeAdapter
from mettle.api_models import (
    MULTI_ROUND_SUITE,
    SUITE_NAMES,
    SessionStatus,
)

logger = logging.getLogger(__name__)

# Redis key prefixes
_PREFIX = "mettle:session"
_RATE_PREFIX = "mettle:rate"

# TTLs in seconds
ACTIVE_SESSION_TTL = 300  # 5 minutes
COMPLETED_SESSION_TTL = 3600  # 1 hour
RATE_LIMIT_WINDOW = 3600  # 1 hour

# Rate limits
MAX_ACTIVE_SESSIONS_PER_USER = 5
MAX_SESSIONS_PER_HOUR = 100


def _key(session_id: str, suffix: str = "") -> str:
    """Build a Redis key."""
    base = f"{_PREFIX}:{session_id}"
    return f"{base}:{suffix}" if suffix else base


def _rate_key(user_id: str, kind: str) -> str:
    return f"{_RATE_PREFIX}:{user_id}:{kind}"


class SessionManager:
    """Manages METTLE verification sessions backed by Redis."""

    def __init__(self, redis_client: Any) -> None:
        self.redis = redis_client

    # ---- Session Lifecycle ----

    async def create_session(
        self,
        user_id: str,
        suites: list[str],
        difficulty: str = "standard",
        entity_id: str | None = None,
        vcp_token: str | None = None,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Create a new verification session.

        Returns (session_id, client_challenges, session_metadata).
        Raises ValueError on rate limit or invalid suites.
        """
        # Rate limiting
        await self._check_rate_limits(user_id)

        # Resolve suite list
        resolved_suites = self._resolve_suites(suites)

        # Generate session ID
        session_id = secrets.token_urlsafe(32)
        now = datetime.now(tz=timezone.utc)
        expires_at = datetime.fromtimestamp(now.timestamp() + ACTIVE_SESSION_TTL, tz=timezone.utc)

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

        for suite in resolved_suites:
            if suite == MULTI_ROUND_SUITE:
                client, server = ChallengeAdapter.generate_novel_reasoning(difficulty)
            elif suite == "intent-provenance" and vcp_token is not None:
                client, server = ChallengeAdapter.generate_intent_provenance(vcp_token=vcp_token)
            else:
                gen = generators.get(suite)
                if gen is None:
                    raise ValueError(f"Unknown suite: {suite}")
                client, server = gen()
            client_challenges[suite] = client
            server_answers[suite] = server

        # Calculate time budget
        has_novel = MULTI_ROUND_SUITE in resolved_suites
        from scripts.engine import NovelReasoningChallenges

        if has_novel:
            params = NovelReasoningChallenges.DIFFICULTY_PARAMS[difficulty]
            time_budget_ms = params["time_budget_s"] * 1000 + len(resolved_suites) * 30000
        else:
            time_budget_ms = len(resolved_suites) * 30000  # 30s per single-shot suite

        # Store session metadata
        session_meta = {
            "session_id": session_id,
            "user_id": user_id,
            "entity_id": entity_id,
            "vcp_token": vcp_token,
            "suites": resolved_suites,
            "difficulty": difficulty,
            "status": SessionStatus.CHALLENGES_GENERATED.value,
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "time_budget_ms": time_budget_ms,
            "start_time": None,
            "current_round": 0,
            "suites_completed": [],
            "suite_results": {},
            "round_data": [],
        }

        # Store in Redis
        pipe = self.redis.pipeline()
        pipe.setex(_key(session_id), ACTIVE_SESSION_TTL, json.dumps(session_meta))
        pipe.setex(_key(session_id, "answers"), ACTIVE_SESSION_TTL, json.dumps(server_answers))
        # Track active sessions for rate limiting
        pipe.sadd(_rate_key(user_id, "active"), session_id)
        pipe.expire(_rate_key(user_id, "active"), ACTIVE_SESSION_TTL)
        # Hourly counter
        pipe.incr(_rate_key(user_id, "hourly"))
        pipe.expire(_rate_key(user_id, "hourly"), RATE_LIMIT_WINDOW)
        await pipe.execute()

        return session_id, client_challenges, session_meta

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get session metadata."""
        raw = await self.redis.get(_key(session_id))
        if raw is None:
            return None
        return json.loads(raw)

    async def get_session_answers(self, session_id: str) -> dict[str, Any] | None:
        """Get server-side answers for a session."""
        raw = await self.redis.get(_key(session_id, "answers"))
        if raw is None:
            return None
        return json.loads(raw)

    async def cancel_session(self, session_id: str, user_id: str) -> bool:
        """Cancel a session. Returns False if not found or wrong user."""
        session = await self.get_session(session_id)
        if session is None:
            return False
        if session["user_id"] != user_id:
            return False
        if session["status"] in (SessionStatus.COMPLETED.value, SessionStatus.CANCELLED.value):
            return False

        session["status"] = SessionStatus.CANCELLED.value
        await self.redis.setex(_key(session_id), COMPLETED_SESSION_TTL, json.dumps(session))
        await self.redis.srem(_rate_key(user_id, "active"), session_id)
        return True

    # ---- Single-Shot Verification ----

    async def verify_single_shot(
        self,
        session_id: str,
        suite: str,
        answers: dict[str, Any],
    ) -> dict[str, Any]:
        """Verify answers for a single-shot suite.

        Returns evaluation results. Raises ValueError on invalid state.
        """
        session = await self.get_session(session_id)
        if session is None:
            raise ValueError("Session not found")

        if session["status"] not in (
            SessionStatus.CHALLENGES_GENERATED.value,
            SessionStatus.IN_PROGRESS.value,
        ):
            raise ValueError(f"Session not in verifiable state: {session['status']}")

        if suite not in session["suites"]:
            raise ValueError(f"Suite '{suite}' not in this session")

        if suite in session["suites_completed"]:
            raise ValueError(f"Suite '{suite}' already completed")

        if suite == MULTI_ROUND_SUITE:
            raise ValueError("Novel reasoning requires multi-round endpoint")

        # Start timing on first verification
        if session["start_time"] is None:
            session["start_time"] = time.time()
            session["status"] = SessionStatus.IN_PROGRESS.value

        # Get server answers and evaluate
        server_answers = await self.get_session_answers(session_id)
        if server_answers is None:
            raise ValueError("Session answers expired")

        suite_server = server_answers.get(suite, {})
        result = ChallengeAdapter.evaluate_single_shot(suite, answers, suite_server)

        # Update session
        session["suites_completed"].append(suite)
        session["suite_results"][suite] = result

        # Check if all suites completed
        if set(session["suites_completed"]) == set(session["suites"]):
            session["status"] = SessionStatus.COMPLETED.value
            ttl = COMPLETED_SESSION_TTL
            # Clean up active session tracking for rate limiting
            await self.redis.srem(_rate_key(session["user_id"], "active"), session_id)
        else:
            ttl = ACTIVE_SESSION_TTL

        await self.redis.setex(_key(session_id), ttl, json.dumps(session))
        return result

    # ---- Multi-Round (Suite 10) ----

    async def submit_round_answer(
        self,
        session_id: str,
        round_num: int,
        answers: dict[str, Any],
    ) -> dict[str, Any]:
        """Submit answers for a multi-round challenge round.

        Returns round evaluation with feedback. Raises ValueError on invalid state.
        """
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

        expected_round = session["current_round"] + 1
        if round_num != expected_round:
            raise ValueError(f"Expected round {expected_round}, got {round_num}")

        # Start timing on first round
        if session["start_time"] is None:
            session["start_time"] = time.time()
            session["status"] = SessionStatus.IN_PROGRESS.value

        # Server-side timing enforcement
        elapsed_ms = (time.time() - session["start_time"]) * 1000

        server_answers = await self.get_session_answers(session_id)
        if server_answers is None:
            raise ValueError("Session answers expired")

        novel_server = server_answers.get(MULTI_ROUND_SUITE, {})
        time_budget_ms = novel_server.get("time_budget_s", 30) * 1000
        num_rounds = novel_server.get("num_rounds", 3)

        time_exceeded = elapsed_ms > time_budget_ms
        time_remaining_ms = max(0, int(time_budget_ms - elapsed_ms))

        # Evaluate each challenge for this round
        round_results: dict[str, Any] = {}
        all_errors: list[str] = []
        total_accuracy = 0.0
        num_challenges = 0

        challenge_answers = answers.get("challenges", answers)
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
            "response_time_ms": round(elapsed_ms, 1),
            "accuracy": round(avg_accuracy, 4),
            "time_exceeded": time_exceeded,
            "results": round_results,
        }
        session["round_data"].append(round_record)
        session["current_round"] = round_num

        # Build feedback
        is_final_round = round_num >= num_rounds
        feedback: dict[str, Any] = {
            "accuracy": round(avg_accuracy, 4),
            "challenge_feedback": round_results,
        }

        # Determine next round data
        next_round_data = None
        if not is_final_round:
            next_round_data = {"round": round_num + 1, "note": "Continue with updated challenge data"}

        if is_final_round:
            # Analyze iteration curve
            curve_result = self._analyze_iteration_curve(session["round_data"], novel_server)
            session["suite_results"][MULTI_ROUND_SUITE] = curve_result
            session["suites_completed"].append(MULTI_ROUND_SUITE)

            # Check if all suites completed
            if set(session["suites_completed"]) == set(session["suites"]):
                session["status"] = SessionStatus.COMPLETED.value
                # Clean up active session tracking for rate limiting
                await self.redis.srem(_rate_key(session["user_id"], "active"), session_id)

        ttl = COMPLETED_SESSION_TTL if session["status"] == SessionStatus.COMPLETED.value else ACTIVE_SESSION_TTL
        await self.redis.setex(_key(session_id), ttl, json.dumps(session))

        return {
            "round_num": round_num,
            "accuracy": round(avg_accuracy, 4),
            "errors": all_errors[:10],
            "feedback": feedback,
            "time_remaining_ms": time_remaining_ms,
            "next_round_data": next_round_data,
        }

    async def get_round_feedback(self, session_id: str, round_num: int) -> dict[str, Any] | None:
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

        elapsed_ms = 0
        if session["start_time"]:
            now = time.time()
            elapsed_ms = int((now - session["start_time"]) * 1000)

        # Check overall pass
        results = session.get("suite_results", {})
        all_passed = all(r.get("passed", False) for r in results.values()) if results else False

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
        }

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
                        - (round_data[round_data.index(rd) - 1]["accuracy"] if round_data.index(rd) > 0 else 0)
                    ),
                    "error_magnitude": 1.0 - rd["accuracy"],
                }
            )

        curve = IterationCurveAnalyzer.analyze_curve(analyzer_rounds)
        pass_threshold = server_answers.get("pass_threshold", 0.65)
        passed = curve["overall"] > pass_threshold and curve["signature"] != "SCRIPT"

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
        # Check active sessions
        active_count = await self.redis.scard(_rate_key(user_id, "active"))
        if active_count is not None and active_count >= MAX_ACTIVE_SESSIONS_PER_USER:
            raise ValueError(
                f"Maximum active sessions ({MAX_ACTIVE_SESSIONS_PER_USER}) exceeded. "
                "Complete or cancel existing sessions."
            )

        # Check hourly limit
        hourly_raw = await self.redis.get(_rate_key(user_id, "hourly"))
        hourly_count = int(hourly_raw) if hourly_raw else 0
        if hourly_count >= MAX_SESSIONS_PER_HOUR:
            raise ValueError(f"Hourly session limit ({MAX_SESSIONS_PER_HOUR}) exceeded. Try again later.")

    # ---- Helpers ----

    @staticmethod
    def _resolve_suites(suites: list[str]) -> list[str]:
        """Resolve 'all' to full suite list and validate names."""
        if "all" in suites:
            return list(SUITE_NAMES)

        invalid = [s for s in suites if s not in SUITE_NAMES]
        if invalid:
            raise ValueError(f"Unknown suites: {invalid}. Valid: {SUITE_NAMES}")

        return suites
