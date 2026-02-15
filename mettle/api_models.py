"""Pydantic models for METTLE v2 API.

Request/response models for session management, verification, and multi-round challenges.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class SessionStatus(str, enum.Enum):
    """Session state machine states."""

    CREATED = "created"
    CHALLENGES_GENERATED = "challenges_generated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


# Suite name constants
SUITE_NAMES = [
    "adversarial",
    "native",
    "self-reference",
    "social",
    "inverse-turing",
    "anti-thrall",
    "agency",
    "counter-coaching",
    "intent-provenance",
    "novel-reasoning",
]

SINGLE_SHOT_SUITES = SUITE_NAMES[:9]
MULTI_ROUND_SUITE = "novel-reasoning"


# ---- Request Models ----


class CreateSessionRequest(BaseModel):
    """Request to start a METTLE verification session."""

    suites: list[str] = Field(default=["all"], description="Suite names or 'all'")
    difficulty: Literal["easy", "standard", "hard"] = "standard"
    entity_id: str | None = Field(default=None, description="Optional entity identifier")


class RoundAnswerRequest(BaseModel):
    """Submit answers for a multi-round challenge round."""

    answers: dict[str, Any] = Field(description="Challenge-specific answers")
    submitted_at: datetime | None = Field(default=None, description="Client-side timestamp")


class VerifyRequest(BaseModel):
    """Submit answers for a single-shot suite."""

    suite: str = Field(description="Suite name to verify")
    answers: dict[str, Any] = Field(description="Suite-specific answers")


# ---- Response Models ----


class CreateSessionResponse(BaseModel):
    """Response after creating a verification session."""

    session_id: str
    created_at: datetime
    expires_at: datetime
    suites: list[str]
    challenges: dict[str, Any] = Field(description="Suite name -> challenge data (no answers)")
    time_budget_ms: int


class RoundFeedbackResponse(BaseModel):
    """Feedback after a multi-round answer submission."""

    round_num: int
    accuracy: float
    errors: list[str]
    feedback: dict[str, Any]
    time_remaining_ms: int
    next_round_data: dict[str, Any] | None = Field(default=None, description="Data for next round; null if final")


class VerifyResponse(BaseModel):
    """Result of a single-shot suite verification."""

    suite: str
    passed: bool
    score: float
    details: dict[str, Any]


class SessionResultResponse(BaseModel):
    """Final results for a completed session."""

    session_id: str
    status: str
    suites_completed: list[str]
    results: dict[str, Any]
    overall_passed: bool
    iteration_curve: dict[str, Any] | None = Field(default=None, description="Only for sessions including Suite 10")
    elapsed_ms: int


class SuiteInfoResponse(BaseModel):
    """Information about a single verification suite."""

    name: str
    display_name: str
    description: str
    suite_number: int
    is_multi_round: bool
    difficulty_levels: list[str]


class SessionStatusResponse(BaseModel):
    """Current status of a verification session."""

    session_id: str
    status: SessionStatus
    suites: list[str]
    created_at: datetime
    expires_at: datetime
    current_round: int | None = None
    suites_completed: list[str] = Field(default_factory=list)
    elapsed_ms: int
