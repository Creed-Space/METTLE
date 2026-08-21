"""METTLE API Router - Machine Evaluation Through Turing-inverse Logic Examination.

Exposes all 12 METTLE verification suites via REST API endpoints.
Suite 10 (Novel Reasoning) supports multi-round sessions with feedback.

SECURITY: All endpoints require authentication. Correct answers are NEVER sent to clients.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status

from mettle.api_models import (
    MULTI_ROUND_SUITE,
    SUITE_NAMES,
    CreateSessionRequest,
    CreateSessionResponse,
    GovernanceAttestation,
    OperatorAttestation,
    PresenceState,
    PresentationChallengeRequest,
    PresentationChallengeResponse,
    PresentationVerifyRequest,
    PresentationVerifyResponse,
    RoundAnswerRequest,
    RoundFeedbackResponse,
    SessionResultResponse,
    SessionStatusResponse,
    SuiteInfoResponse,
    VerifyRequest,
    VerifyResponse,
)
from mettle.auth import AuthenticatedUser, require_authenticated_user
from mettle.challenge_adapter import SUITE_REGISTRY
from mettle.llm_challenges import is_available as llm_challenges_available
from mettle.presence import issuer_signed_session_presence
from mettle.session_manager import SessionManager, SessionRateLimitError
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mettle", tags=["mettle"])

_OPERATOR_COMMITMENT_MAX_AGE_SECONDS = 5 * 60
_OPERATOR_COMMITMENT_FUTURE_SKEW_SECONDS = 30
_OPERATOR_NONCE_TTL_SECONDS = (
    _OPERATOR_COMMITMENT_MAX_AGE_SECONDS + _OPERATOR_COMMITMENT_FUTURE_SKEW_SECONDS
)
_OPERATOR_NONCE_PREFIX = "mettle:operator-commitment:nonce:"
_RELEASE_OPERATOR_NONCE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""

# Type alias for auth dependency
AuthUser = Annotated[AuthenticatedUser, Depends(require_authenticated_user)]


async def get_session_manager(request: Request) -> SessionManager:
    """Get a SessionManager instance with Redis client.

    Used as a FastAPI dependency for testability.
    """
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="METTLE API requires Redis for session management",
        )
    return SessionManager(redis)


# Type alias for session manager dependency
MettleManager = Annotated[SessionManager, Depends(get_session_manager)]


# ---- Suite Information ----


@router.get("/suites", response_model=list[SuiteInfoResponse])
async def list_suites(_user: AuthUser) -> list[SuiteInfoResponse]:
    """List all available verification suites."""
    suites = []
    for name, (display_name, description, suite_num) in SUITE_REGISTRY.items():
        # llm-dynamic requires API key + anthropic package
        available = llm_challenges_available() if name == "llm-dynamic" else True
        suites.append(
            SuiteInfoResponse(
                name=name,
                display_name=display_name,
                description=description,
                suite_number=suite_num,
                is_multi_round=name == MULTI_ROUND_SUITE,
                difficulty_levels=["easy", "standard", "hard"],
                available=available,
            )
        )
    return suites


@router.get("/suites/{suite_name}", response_model=SuiteInfoResponse)
async def get_suite_info(
    _user: AuthUser, suite_name: str = Path(description="Suite name")
) -> SuiteInfoResponse:
    """Get information about a specific suite."""
    if suite_name not in SUITE_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite not found: {suite_name}. Valid suites: {SUITE_NAMES}",
        )

    display_name, description, suite_num = SUITE_REGISTRY[suite_name]
    available = llm_challenges_available() if suite_name == "llm-dynamic" else True
    return SuiteInfoResponse(
        name=suite_name,
        display_name=display_name,
        description=description,
        suite_number=suite_num,
        is_multi_round=suite_name == MULTI_ROUND_SUITE,
        difficulty_levels=["easy", "standard", "hard"],
        available=available,
    )


# ---- Session Management ----


@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "description": "Active-session or hourly creation quota exceeded"
        }
    },
)
async def create_session(
    request: CreateSessionRequest, user: AuthUser, mgr: MettleManager
) -> CreateSessionResponse:
    """Start a new METTLE verification session.

    Generates challenges for the requested suites. Challenge data is returned
    WITHOUT correct answers -- the server stores answers for secure evaluation.
    """
    try:
        operator_commitment = (
            request.operator_commitment.model_dump(mode="json")
            if request.operator_commitment
            else None
        )
        nonce_reservation: tuple[str, str] | None = None
        if operator_commitment is not None:
            if (
                _verify_operator_commitment(
                    operator_commitment,
                    request.entity_id,
                )
                is None
            ):
                raise ValueError("Operator commitment is invalid or stale")
            nonce_key = _OPERATOR_NONCE_PREFIX + operator_commitment["nonce"]
            reservation_token = secrets.token_urlsafe(32)
            reserved = await mgr.redis.set(
                nonce_key,
                reservation_token,
                ex=_OPERATOR_NONCE_TTL_SECONDS,
                nx=True,
            )
            if not reserved:
                raise ValueError("Operator commitment nonce has already been used")
            nonce_reservation = (nonce_key, reservation_token)

        try:
            session_id, challenges, meta = await mgr.create_session(
                user_id=user.user_id,
                suites=request.suites,
                difficulty=request.difficulty,
                entity_id=request.entity_id,
                vcp_token=request.vcp_token,
                operator_commitment=operator_commitment,
                presence=request.presence.model_dump(mode="json")
                if request.presence
                else None,
            )
        except Exception:
            if nonce_reservation is not None:
                await _release_operator_nonce(mgr.redis, *nonce_reservation)
            raise

        logger.info(
            "METTLE session created",
            extra={
                "session_id": session_id,
                "user_id": user.user_id,
                "suites": meta["suites"],
                "difficulty": request.difficulty,
            },
        )

        return CreateSessionResponse(
            session_id=session_id,
            created_at=datetime.fromisoformat(meta["created_at"]),
            expires_at=datetime.fromisoformat(meta["expires_at"]),
            suites=meta["suites"],
            challenges=challenges,
            time_budget_ms=meta["time_budget_ms"],
            presence=(
                PresenceState.model_validate(
                    issuer_signed_session_presence(
                        meta.get("presence"), session_id=session_id
                    )
                )
                if meta.get("presence")
                else None
            ),
        )

    except RedisError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="METTLE session storage temporarily unavailable",
        ) from e
    except SessionRateLimitError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
            headers={"Retry-After": str(e.retry_after)},
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    except Exception as e:
        logger.error(f"Failed to create METTLE session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session",
        ) from e


@router.get("/sessions/{session_id}", response_model=SessionStatusResponse)
async def get_session_status(
    user: AuthUser, mgr: MettleManager, session_id: str = Path(description="Session ID")
) -> SessionStatusResponse:
    """Get current status of a verification session."""
    session = await mgr.get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or expired"
        )

    if session["user_id"] != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not your session"
        )

    import time

    elapsed_ms = 0
    if session.get("start_time"):
        elapsed_ms = int((time.time() - session["start_time"]) * 1000)

    return SessionStatusResponse(
        session_id=session_id,
        status=session["status"],
        suites=session["suites"],
        created_at=datetime.fromisoformat(session["created_at"]),
        expires_at=datetime.fromisoformat(session["expires_at"]),
        current_round=session.get("current_round"),
        suites_completed=session.get("suites_completed", []),
        presence=(
            PresenceState.model_validate(
                issuer_signed_session_presence(
                    session.get("presence"),
                    session_id=session_id,
                    completed=session["status"] == "completed",
                )
            )
            if session.get("presence")
            else None
        ),
        elapsed_ms=elapsed_ms,
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_session(
    user: AuthUser, mgr: MettleManager, session_id: str = Path(description="Session ID")
) -> None:
    """Cancel an active verification session."""
    success = await mgr.cancel_session(session_id, user.user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found, already completed, or not yours",
        )

    logger.info(
        "METTLE session cancelled",
        extra={"session_id": session_id, "user_id": user.user_id},
    )


# ---- Single-Shot Verification (Suites 1-9) ----


@router.post("/sessions/{session_id}/verify", response_model=VerifyResponse)
async def verify_single_shot(
    request: VerifyRequest,
    user: AuthUser,
    mgr: MettleManager,
    session_id: str = Path(description="Session ID"),
) -> VerifyResponse:
    """Submit answers for a single-shot suite (Suites 1-9).

    Evaluates the submitted answers against server-stored correct answers.
    """
    try:
        # Verify session ownership
        session = await mgr.get_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or expired",
            )
        if session["user_id"] != user.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Not your session"
            )

        result = await mgr.verify_single_shot(
            session_id,
            request.suite,
            request.answers,
            request.presence_proof.model_dump() if request.presence_proof else None,
        )

        logger.info(
            "METTLE suite verified",
            extra={
                "session_id": session_id,
                "suite": request.suite,
                "passed": result["passed"],
                "score": result["score"],
            },
        )

        return VerifyResponse(
            suite=request.suite,
            passed=result["passed"],
            score=result["score"],
            details=result["details"],
            presence=result.get("presence"),
            next_challenge=result.get("next_challenge"),
        )

    except RedisError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="METTLE session storage temporarily unavailable",
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"METTLE verification failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Verification failed",
        ) from e


# ---- Multi-Round (Suite 10: Novel Reasoning) ----


@router.post(
    "/sessions/{session_id}/rounds/{round_num}/answer",
    response_model=RoundFeedbackResponse,
)
async def submit_round_answer(
    request: RoundAnswerRequest,
    user: AuthUser,
    mgr: MettleManager,
    session_id: str = Path(description="Session ID"),
    round_num: int = Path(ge=1, le=5, description="Round number (1-based)"),
) -> RoundFeedbackResponse:
    """Submit answers for a multi-round challenge round (Suite 10).

    After each round, feedback is provided including accuracy and errors.
    The next round's data is included for progressive disclosure.
    """
    try:
        # Verify session ownership
        session = await mgr.get_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or expired",
            )
        if session["user_id"] != user.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Not your session"
            )

        result = await mgr.submit_round_answer(
            session_id,
            round_num,
            request.answers,
            request.presence_proof.model_dump() if request.presence_proof else None,
        )

        return RoundFeedbackResponse(
            round_num=result["round_num"],
            accuracy=result["accuracy"],
            errors=result["errors"],
            feedback=result["feedback"],
            time_remaining_ms=result["time_remaining_ms"],
            next_round_data=result.get("next_round_data"),
            presence=result.get("presence"),
            next_challenge=result.get("next_challenge"),
        )

    except RedisError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="METTLE session storage temporarily unavailable",
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"METTLE round submission failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Round submission failed",
        ) from e


@router.get("/sessions/{session_id}/rounds/{round_num}/feedback")
async def get_round_feedback(
    user: AuthUser,
    mgr: MettleManager,
    session_id: str = Path(description="Session ID"),
    round_num: int = Path(ge=1, le=5, description="Round number"),
) -> dict[str, Any]:
    """Get feedback for a completed round."""

    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or expired"
        )
    if session["user_id"] != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not your session"
        )

    feedback = await mgr.get_round_feedback(session_id, round_num)
    if feedback is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Round {round_num} not yet completed",
        )

    return feedback


# ---- Results ----


@router.get("/sessions/{session_id}/result", response_model=SessionResultResponse)
async def get_session_result(
    user: AuthUser,
    mgr: MettleManager,
    session_id: str = Path(description="Session ID"),
    include_vcp: bool = Query(
        default=False, description="Include VCP-compatible attestation in response"
    ),
) -> SessionResultResponse:
    """Get final results for a completed session.

    Returns 404 if session not completed yet.
    When include_vcp=true, includes a signed credential for a tier-qualifying
    result, or an unsigned evidence receipt when no tier was earned.
    """

    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or expired"
        )
    if session["user_id"] != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not your session"
        )

    result = await mgr.get_result(session_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session not completed. Status: {session['status']}",
        )

    from mettle.vcp import compute_tier

    suite_results = result.get("results", {})
    suites_passed = [s for s, r in suite_results.items() if r.get("passed", False)]
    suites_failed = [s for s, r in suite_results.items() if not r.get("passed", False)]
    tier = compute_tier(suites_passed)
    result["verified"] = bool(result.get("overall_passed", False))
    result["assurance"] = "mettle_behavioral_verification"
    result["credential_eligible"] = tier != "none"
    result["tier"] = tier

    # Build VCP attestation if requested
    vcp_attestation = None
    if include_vcp:
        from mettle.vcp import build_mettle_attestation

        issuer_key_id = "mettle-vcp-v1"
        try:
            from mettle.signing import get_public_key_info

            discovered_key_id = get_public_key_info().get("key_id")
            if isinstance(discovered_key_id, str) and discovered_key_id:
                issuer_key_id = discovered_key_id
        except ImportError:
            pass
        pass_rate = sum(
            1 for r in suite_results.values() if r.get("passed", False)
        ) / max(len(suite_results), 1)
        vcp_attestation = build_mettle_attestation(
            session_id=session_id,
            difficulty=session.get("difficulty", "standard"),
            suites_passed=suites_passed,
            suites_failed=suites_failed,
            pass_rate=pass_rate,
            subject_id=user.user_id,
            entity_id=session.get("entity_id"),
            key_id=issuer_key_id,
            presence=session.get("presence"),
        )

    result["vcp_attestation"] = vcp_attestation

    # Build GovernanceAttestation from VCP token if present
    governance_attestation = None
    vcp_token = session.get("vcp_token")
    if vcp_token:
        governance_attestation = _build_governance_attestation(
            vcp_token,
            entity_id=session.get("entity_id"),
            session_id=session_id,
            tier=tier,
        )
    result["governance_attestation"] = governance_attestation

    # Build OperatorAttestation from operator commitment if present
    operator_attestation = None
    operator_commitment = session.get("operator_commitment")
    if operator_commitment:
        accepted_at_raw = session.get("created_at")
        try:
            accepted_at = (
                datetime.fromisoformat(accepted_at_raw)
                if isinstance(accepted_at_raw, str)
                else None
            )
        except ValueError:
            accepted_at = None
        operator_attestation = _build_operator_attestation(
            operator_commitment,
            session.get("entity_id"),
            accepted_at=accepted_at,
        )
    result["operator_attestation"] = operator_attestation

    return SessionResultResponse(**result)


@router.post(
    "/presentation-challenges",
    response_model=PresentationChallengeResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_presentation_challenge(
    request: PresentationChallengeRequest,
    user: AuthUser,
    mgr: MettleManager,
) -> PresentationChallengeResponse:
    """Create a fresh verifier-owned challenge for a bound credential."""
    try:
        challenge = await mgr.create_presentation_challenge(
            verifier_user_id=user.user_id,
            credential_jti=request.credential_jti,
            audience=request.audience,
        )
        return PresentationChallengeResponse(**challenge)
    except SessionRateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    except RedisError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="METTLE presentation storage temporarily unavailable",
        ) from exc


@router.post(
    "/presentations/verify",
    response_model=PresentationVerifyResponse,
)
async def verify_credential_presentation(
    request: PresentationVerifyRequest,
    raw_request: Request,
    user: AuthUser,
    mgr: MettleManager,
) -> PresentationVerifyResponse:
    """Verify issuer integrity, current policy, and live holder possession."""
    from mettle.signing import get_public_key_pem
    from mettle.vcp import verify_mettle_attestation

    issuer_public_key = get_public_key_pem()
    if issuer_public_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="METTLE credential verification is unavailable",
        )
    try:
        attestation_valid = verify_mettle_attestation(
            request.attestation, issuer_public_key
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        attestation_valid = False
    if not attestation_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Credential signature, policy, or expiry is invalid",
        )

    metadata = request.attestation.get("metadata")
    if not isinstance(metadata, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Credential metadata is invalid",
        )
    proof = metadata.get("proof_of_possession")
    if not isinstance(proof, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Credential is not bound to a holder key",
        )
    required_metadata = ("jti", "audience", "tier", "subject_id")
    if not all(isinstance(metadata.get(field), str) for field in required_metadata):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Credential metadata is invalid",
        )
    required_proof = ("public_key_pem", "key_fingerprint", "transcript_hash")
    if not all(isinstance(proof.get(field), str) for field in required_proof):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Credential holder proof is invalid",
        )
    jti = metadata["jti"]
    checker = getattr(raw_request.app.state, "credential_revocation_checker", None)
    if not callable(checker):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Credential revocation service is unavailable",
        )
    try:
        revoked = bool(checker(jti))
    except Exception as exc:
        logger.error("Credential revocation check failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Credential revocation service is unavailable",
        ) from exc
    if revoked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Credential has been revoked",
        )

    try:
        await mgr.verify_presentation(
            verifier_user_id=user.user_id,
            challenge_id=request.challenge_id,
            credential_jti=jti,
            audience=metadata["audience"],
            public_key_pem=proof["public_key_pem"],
            holder_signature=request.holder_signature,
        )
    except RedisError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="METTLE presentation storage temporarily unavailable",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return PresentationVerifyResponse(
        credential_jti=jti,
        audience=metadata["audience"],
        tier=metadata["tier"],
        subject_id=metadata["subject_id"],
        entity_id=metadata.get("entity_id"),
        key_fingerprint=proof["key_fingerprint"],
        transcript_hash=proof["transcript_hash"],
    )


@router.get("/.well-known/vcp-keys")
async def get_vcp_keys() -> dict:
    """Serve public key for VCP attestation signature verification.

    This endpoint enables trust config discovery for VCP consumers.
    """
    try:
        from mettle.signing import get_public_key_info

        return get_public_key_info()
    except ImportError:
        return {
            "key_id": "mettle-vcp-v1",
            "algorithm": "Ed25519",
            "public_key_pem": None,
            "available": False,
            "error": "cryptography package not installed",
        }


# ---- Attestation Builders ----


def _build_governance_attestation(
    vcp_token: str,
    *,
    entity_id: str | None,
    session_id: str,
    tier: str,
) -> GovernanceAttestation | None:
    """Build GovernanceAttestation from a VCP token.

    Parses caller-supplied CSM-1 metadata without promoting it into proof.
    """
    import hashlib

    try:
        from mettle.vcp import parse_csm1_token

        parsed = parse_csm1_token(vcp_token)
    except (ValueError, TypeError, AttributeError, UnicodeError, ImportError):
        logger.warning("Failed to parse VCP token for governance attestation")
        return None

    # Determine framework from constitution ID
    constitution_id = parsed.constitution_id or ""
    framework = "none"
    if "creed" in constitution_id.lower() or parsed.extra_lines.get("F"):
        framework = "creed-space"
    elif constitution_id:
        framework = "custom"

    # Hash the constitution reference for integrity
    constitutional_hash = None
    if parsed.constitution_ref:
        constitutional_hash = hashlib.sha256(
            parsed.constitution_ref.encode()
        ).hexdigest()

    # Raw caller metadata has no cryptographic provenance. Environment switches
    # and digest allowlists cannot turn self-asserted text into governance proof.
    now = datetime.now(tz=timezone.utc)
    expires_at = now + timedelta(hours=1)

    source_vcp_hash = hashlib.sha256(vcp_token.encode()).hexdigest()
    return GovernanceAttestation(
        entity_id=entity_id,
        session_id=session_id,
        tier=tier,
        source_vcp_hash=source_vcp_hash,
        source_verified=False,
        framework=framework,
        framework_version=parsed.constitution_version,
        constitutional_hash=constitutional_hash,
        has_action_gate=False,
        has_drift_detection=False,
        has_bilateral=False,
        observed_at=now,
        expires_at=expires_at,
        attestation_signature=None,
    )


def _parse_operator_issued_at(value: Any) -> datetime | None:
    """Parse the signed timestamp without silently accepting a non-UTC value."""

    if isinstance(value, datetime):
        issued_at = value
    elif isinstance(value, str):
        try:
            issued_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if issued_at.tzinfo is None or issued_at.utcoffset() != timedelta(0):
        return None
    return issued_at.astimezone(timezone.utc)


def _canonical_operator_commitment(
    commitment: dict[str, Any], entity_id: str, issued_at: datetime
) -> bytes:
    """Return the canonical version-1 message signed by the operator."""

    issued_at_text = issued_at.isoformat().replace("+00:00", "Z")
    return json.dumps(
        {
            "contact_hash": commitment["contact_hash"],
            "contact_method": commitment["contact_method"],
            "entity_id": entity_id,
            "issued_at": issued_at_text,
            "nonce": commitment["nonce"],
            "operator_pseudonym": commitment["operator_pseudonym"],
            "operator_public_key": commitment["operator_public_key"],
            "version": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _verify_operator_commitment(
    commitment: dict[str, Any],
    entity_id: str | None,
    *,
    accepted_at: datetime | None = None,
) -> datetime | None:
    """Verify one fresh, fully bound operator commitment.

    ``accepted_at`` is the authoritative receipt time.  Passing the session's
    persisted creation time allows later result reads to revalidate the original
    acceptance decision without making an old commitment look freshly issued.
    """

    if not entity_id:
        logger.warning("Operator commitment requires a non-empty entity_id")
        return None

    required_fields = {
        "operator_pseudonym",
        "operator_public_key",
        "signed_commitment",
        "contact_method",
        "contact_hash",
        "issued_at",
        "nonce",
    }
    if set(commitment) != required_fields or not all(
        commitment.get(field) for field in required_fields
    ):
        logger.warning("Operator commitment missing required fields")
        return None

    if not isinstance(commitment["operator_pseudonym"], str) or not isinstance(
        commitment["operator_public_key"], str
    ):
        return None
    if commitment["contact_method"] not in {
        "email_hash",
        "platform_handle",
        "legal_entity",
    }:
        return None
    if (
        not isinstance(commitment["contact_hash"], str)
        or re.fullmatch(r"[0-9a-f]{64}", commitment["contact_hash"]) is None
    ):
        return None
    if (
        not isinstance(commitment["nonce"], str)
        or re.fullmatch(r"[0-9a-f]{64}", commitment["nonce"]) is None
    ):
        return None

    issued_at = _parse_operator_issued_at(commitment["issued_at"])
    reference = accepted_at or datetime.now(timezone.utc)
    if issued_at is None or reference.tzinfo is None or reference.utcoffset() is None:
        logger.warning("Operator commitment has an invalid timestamp")
        return None
    reference = reference.astimezone(timezone.utc)
    age_seconds = (reference - issued_at).total_seconds()
    if age_seconds > _OPERATOR_COMMITMENT_MAX_AGE_SECONDS:
        logger.warning("Operator commitment is stale")
        return None
    if age_seconds < -_OPERATOR_COMMITMENT_FUTURE_SKEW_SECONDS:
        logger.warning("Operator commitment timestamp is too far in the future")
        return None

    # Verify Ed25519 signature
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import load_pem_public_key

        public_key = load_pem_public_key(commitment["operator_public_key"].encode())
        if not isinstance(public_key, Ed25519PublicKey):
            logger.warning("Operator public key is not Ed25519")
            return None

        signature_bytes = base64.b64decode(
            commitment["signed_commitment"], validate=True
        )
        if len(signature_bytes) != 64:
            return None
        public_key.verify(
            signature_bytes,
            _canonical_operator_commitment(commitment, entity_id, issued_at),
        )

    except ImportError:
        logger.warning(
            "cryptography package not available for operator signature verification"
        )
        return None
    except Exception:
        logger.warning(
            "Operator commitment signature verification failed", exc_info=True
        )
        return None

    return issued_at


async def _release_operator_nonce(
    redis: Any,
    nonce_key: str,
    reservation_token: str,
) -> None:
    """Release only the nonce reservation created by this request."""

    await redis.eval(
        _RELEASE_OPERATOR_NONCE_SCRIPT,
        1,
        nonce_key,
        reservation_token,
    )


def _build_operator_attestation(
    commitment: dict[str, Any],
    entity_id: str | None,
    *,
    accepted_at: datetime | None,
) -> OperatorAttestation | None:
    """Build an attestation from a signature valid at its persisted receipt time."""

    if accepted_at is None:
        logger.warning("Operator commitment receipt time is unavailable")
        return None

    issued_at = _verify_operator_commitment(
        commitment,
        entity_id,
        accepted_at=accepted_at,
    )
    if issued_at is None:
        return None

    return OperatorAttestation(
        operator_pseudonym=commitment["operator_pseudonym"],
        operator_public_key=commitment["operator_public_key"],
        operator_signed_commitment=commitment["signed_commitment"],
        commitment_timestamp=issued_at,
        commitment_nonce=commitment["nonce"],
        contact_method=commitment["contact_method"],
        contact_hash=commitment["contact_hash"],
    )
