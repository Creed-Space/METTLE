"""METTLE API Router - Machine Evaluation Through Turing-inverse Logic Examination.

Exposes all 12 METTLE verification suites via REST API endpoints.
Suite 10 (Novel Reasoning) supports multi-round sessions with feedback.

SECURITY: All endpoints require authentication. Correct answers are NEVER sent to clients.
"""

import logging
from inspect import isawaitable
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Path,
    Query,
    Request,
    status,
)

from mettle.api_models import (
    MULTI_ROUND_SUITE,
    SUITE_NAMES,
    CredentialStatusRequest,
    CredentialStatusResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    GovernanceAttestation,
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
from mettle.app_config import settings
from mettle.challenge_adapter import SUITE_REGISTRY
from mettle.llm_challenges import is_available as llm_challenges_available
from mettle.presence import issuer_signed_session_presence
from mettle.rate_limit import CREDENTIAL_STATUS_RATE_LIMIT, limiter
from mettle.session_manager import SessionManager, SessionRateLimitError
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mettle", tags=["mettle"])

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


async def _require_credential_issuance_dependencies(request: Request) -> None:
    """Fail closed when the embedding service reports an unhealthy authority."""
    guard = getattr(request.app.state, "credential_issuance_guard", None)
    if guard is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Credential issuance dependencies are unavailable",
        )
    if not callable(guard):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Credential issuance dependencies are unavailable",
        )
    try:
        ready = guard()
        if isawaitable(ready):
            ready = await ready
    except Exception as exc:
        logger.warning(
            "credential_issuance_guard_failed", extra={"error": type(exc).__name__}
        )
        ready = False
    if ready is not True:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Credential issuance dependencies are unavailable",
        )


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
        session_id, challenges, meta = await mgr.create_session(
            user_id=user.user_id,
            suites=request.suites,
            difficulty=request.difficulty,
            entity_id=request.entity_id,
            vcp_token=request.vcp_token,
            presence=request.presence.model_dump() if request.presence else None,
            allow_third_party_llm=request.allow_third_party_llm,
        )

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


# ---- Credential status ----


@router.post(
    "/credentials/status",
    response_model=CredentialStatusResponse,
    summary="Get authenticated credential status",
)
@limiter.limit(CREDENTIAL_STATUS_RATE_LIMIT)
async def get_credential_status(
    request: Request,
    body: CredentialStatusRequest = Body(...),
) -> CredentialStatusResponse:
    """Return a short-lived signed good or revoked status receipt."""
    checker = getattr(request.app.state, "credential_revocation_checker", None)
    if not callable(checker):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Credential revocation service is unavailable",
        )
    try:
        revoked = bool(checker(body.credential_jti))
        from mettle.signing import get_public_key_info
        from mettle.vcp import build_credential_status_receipt

        key_id = get_public_key_info().get("key_id")
        if not isinstance(key_id, str) or not key_id:
            raise RuntimeError("Credential status key is unavailable")
        receipt = build_credential_status_receipt(
            body.credential_jti,
            revoked=revoked,
            key_id=key_id,
        )
    except Exception as exc:
        logger.error("Credential status lookup failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Credential revocation service is unavailable",
        ) from exc
    return CredentialStatusResponse(**receipt)


# ---- Results ----


@router.get("/sessions/{session_id}/result", response_model=SessionResultResponse)
async def get_session_result(
    user: AuthUser,
    mgr: MettleManager,
    raw_request: Request,
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
    credential_suites_passed = [
        s
        for s, r in suite_results.items()
        if r.get("passed", False) and r.get("credential_eligible") is True
    ]
    supplemental_suites_passed = sorted(
        set(suites_passed) - set(credential_suites_passed)
    )
    suites_failed = [s for s, r in suite_results.items() if not r.get("passed", False)]
    tier = compute_tier(credential_suites_passed)
    result["verified"] = bool(result.get("overall_passed", False))
    result["assurance"] = "mettle_behavioral_verification"
    result["credential_eligible"] = tier != "none"
    result["credential_suites_passed"] = sorted(credential_suites_passed)
    result["supplemental_suites_passed"] = supplemental_suites_passed
    result["tier"] = tier

    # Build VCP attestation if requested
    vcp_attestation = None
    if include_vcp:
        if not settings.credential_issuance_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Credential issuance is temporarily disabled",
            )
        await _require_credential_issuance_dependencies(raw_request)
        from mettle.vcp import build_mettle_attestation

        vcp_attestation = await mgr.get_cached_credential(session_id)
        if vcp_attestation is None:
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
            completed_at_raw = session.get("completed_at")
            if isinstance(completed_at_raw, str):
                reviewed_at = datetime.fromisoformat(completed_at_raw)
            else:
                # Historical in-flight sessions may predate completed_at. Use
                # their challenge issue time, never retrieval time, so delayed
                # reads cannot reset evidence freshness.
                reviewed_at = datetime.fromtimestamp(
                    float(session["start_time"]), tz=timezone.utc
                )
            candidate = build_mettle_attestation(
                session_id=session_id,
                difficulty=session.get("difficulty", "standard"),
                suites_passed=credential_suites_passed,
                suites_failed=suites_failed,
                supplemental_suites_passed=supplemental_suites_passed,
                pass_rate=pass_rate,
                subject_id=user.user_id,
                entity_id=session.get("entity_id"),
                key_id=issuer_key_id,
                presence=session.get("presence"),
                reviewed_at=reviewed_at,
            )
            vcp_attestation = (
                await mgr.cache_credential_once(session_id, candidate)
                if candidate.get("credential_issued") is True
                else candidate
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
    from mettle.signing import get_public_keyring
    from mettle.vcp import verify_mettle_attestation_with_keyring

    issuer_keyring = get_public_keyring()
    if not issuer_keyring:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="METTLE credential verification is unavailable",
        )
    if not verify_mettle_attestation_with_keyring(request.attestation, issuer_keyring):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Credential signature, policy, or expiry is invalid",
        )

    metadata = request.attestation["metadata"]
    proof = metadata.get("proof_of_possession")
    if not isinstance(proof, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Credential is not bound to a holder key",
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

    # Recheck after the one-time challenge is validated and consumed. Revocation
    # is a separate durable authority, so the earlier check alone leaves a race
    # in which a credential can be revoked while holder proof is being verified.
    try:
        revoked_after_verification = bool(checker(jti))
    except Exception as exc:
        logger.error("Credential revocation recheck failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Credential revocation service is unavailable",
        ) from exc
    if revoked_after_verification:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Credential has been revoked",
        )

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
    except (ValueError, ImportError):
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
