"""METTLE v2 API Router - Machine Evaluation Through Turing-inverse Logic Examination.

Exposes all 10 METTLE verification suites via REST API endpoints.
Suite 10 (Novel Reasoning) supports multi-round sessions with feedback.

SECURITY: All endpoints require authentication. Correct answers are NEVER sent to clients.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status

from mettle.auth import AuthenticatedUser, require_authenticated_user
from mettle.challenge_adapter import SUITE_REGISTRY
from mettle.api_models import (
    MULTI_ROUND_SUITE,
    SUITE_NAMES,
    CreateSessionRequest,
    CreateSessionResponse,
    RoundAnswerRequest,
    RoundFeedbackResponse,
    SessionResultResponse,
    SessionStatusResponse,
    SuiteInfoResponse,
    VerifyRequest,
    VerifyResponse,
)
from mettle.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/mettle", tags=["mettle"])

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
        suites.append(
            SuiteInfoResponse(
                name=name,
                display_name=display_name,
                description=description,
                suite_number=suite_num,
                is_multi_round=name == MULTI_ROUND_SUITE,
                difficulty_levels=["easy", "standard", "hard"],
            )
        )
    return suites


@router.get("/suites/{suite_name}", response_model=SuiteInfoResponse)
async def get_suite_info(_user: AuthUser, suite_name: str = Path(description="Suite name")) -> SuiteInfoResponse:
    """Get information about a specific suite."""
    if suite_name not in SUITE_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suite not found: {suite_name}. Valid suites: {SUITE_NAMES}",
        )

    display_name, description, suite_num = SUITE_REGISTRY[suite_name]
    return SuiteInfoResponse(
        name=suite_name,
        display_name=display_name,
        description=description,
        suite_number=suite_num,
        is_multi_round=suite_name == MULTI_ROUND_SUITE,
        difficulty_levels=["easy", "standard", "hard"],
    )


# ---- Session Management ----


@router.post("/sessions", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(request: CreateSessionRequest, user: AuthUser, mgr: MettleManager) -> CreateSessionResponse:
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
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or expired")

    if session["user_id"] != user.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your session")

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
        elapsed_ms=elapsed_ms,
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_session(user: AuthUser, mgr: MettleManager, session_id: str = Path(description="Session ID")) -> None:
    """Cancel an active verification session."""
    success = await mgr.cancel_session(session_id, user.user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found, already completed, or not yours",
        )

    logger.info("METTLE session cancelled", extra={"session_id": session_id, "user_id": user.user_id})


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
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or expired")
        if session["user_id"] != user.user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your session")

        result = await mgr.verify_single_shot(session_id, request.suite, request.answers)

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
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"METTLE verification failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Verification failed",
        ) from e


# ---- Multi-Round (Suite 10: Novel Reasoning) ----


@router.post("/sessions/{session_id}/rounds/{round_num}/answer", response_model=RoundFeedbackResponse)
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
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or expired")
        if session["user_id"] != user.user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your session")

        result = await mgr.submit_round_answer(session_id, round_num, request.answers)

        return RoundFeedbackResponse(
            round_num=result["round_num"],
            accuracy=result["accuracy"],
            errors=result["errors"],
            feedback=result["feedback"],
            time_remaining_ms=result["time_remaining_ms"],
            next_round_data=result.get("next_round_data"),
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or expired")
    if session["user_id"] != user.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your session")

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
) -> SessionResultResponse:
    """Get final results for a completed session.

    Returns 404 if session not completed yet.
    """

    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or expired")
    if session["user_id"] != user.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your session")

    result = await mgr.get_result(session_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session not completed. Status: {session['status']}",
        )

    return SessionResultResponse(**result)
