"""Pydantic models for METTLE API.

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
    "governance",  # Suite 11: Governance verification (action gates, constitutional recitation, etc.)
    "llm-dynamic",  # Suite 12: Claude-powered dynamic challenges (requires ANTHROPIC_API_KEY)
]

MULTI_ROUND_SUITE = "novel-reasoning"
GOVERNANCE_SUITE = "governance"
LLM_DYNAMIC_SUITE = "llm-dynamic"
SINGLE_SHOT_SUITES = [s for s in SUITE_NAMES if s != MULTI_ROUND_SUITE]


# ---- Request Models ----


class OperatorCommitment(BaseModel):
    """Operator accountability commitment submitted with session creation.

    The operator signs a commitment accepting accountability for the agent.
    Ed25519 signature is verified server-side before attestation is issued.
    """

    operator_pseudonym: str = Field(description="Operator identifier (can be pseudonymous)")
    operator_public_key: str = Field(description="Ed25519 public key (PEM format)")
    signed_commitment: str = Field(
        description="Base64-encoded Ed25519 signature over: 'I accept accountability for agent {entity_id}'"
    )
    contact_method: str = Field(description="Contact method type: email_hash, platform_handle, legal_entity")
    contact_hash: str = Field(description="SHA-256 of actual contact info (verifiable without revealing)")


class CreateSessionRequest(BaseModel):
    """Request to start a METTLE verification session."""

    suites: list[str] = Field(default=["all"], description="Suite names or 'all'")
    difficulty: Literal["easy", "standard", "hard"] = "standard"
    entity_id: str | None = Field(default=None, description="Optional entity identifier")
    vcp_token: str | None = Field(
        default=None,
        description="Optional CSM-1 VCP token for enhanced Suite 9 verification",
    )
    operator_commitment: OperatorCommitment | None = Field(
        default=None,
        description="Optional operator accountability commitment (enables Platinum tier)",
    )


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


class GovernanceAttestation(BaseModel):
    """Attests the governance framework governing an agent.

    Populated during METTLE verification when the agent provides a VCP token
    containing Creed governance metadata. Enables platforms to distinguish
    between governed and ungoverned agents.

    Trust tier implications:
    - Platinum requires governance_attestation to be present and verified
    - has_action_gate is the key differentiator for the Rathbun scenario
    """

    framework: str = Field(description="Governance framework: creed-space, custom, none")
    framework_version: str | None = Field(default=None, description="Framework version (e.g. 2.1.0)")
    constitutional_hash: str | None = Field(
        default=None,
        description="SHA-256 hash of active constitution at verification time",
    )
    has_action_gate: bool = Field(
        default=False,
        description="Whether agent has action-level governance (Public Action Gate or equivalent)",
    )
    has_drift_detection: bool = Field(
        default=False,
        description="Whether constitution drift is monitored at runtime",
    )
    has_bilateral: bool = Field(
        default=False,
        description="Whether bilateral alignment is active",
    )
    verified_at: datetime = Field(description="When governance was verified")
    attestation_signature: str | None = Field(
        default=None,
        description="Ed25519 signature over governance fields",
    )


class OperatorAttestation(BaseModel):
    """Cryptographic link from agent to operator.

    Even pseudonymous operators provide a verifiable accountability chain.
    The contact_hash allows platforms to verify contact info exists without
    revealing it publicly. If the agent causes harm, the platform can request
    the operator reveal themselves by providing the preimage.
    """

    operator_pseudonym: str = Field(description="Operator identifier (can be pseudonymous)")
    operator_public_key: str = Field(description="Ed25519 public key (PEM format)")
    operator_signed_commitment: str = Field(
        description="Operator signs: 'I accept accountability for agent {entity_id}'"
    )
    commitment_timestamp: datetime = Field(description="When commitment was signed")
    contact_method: str = Field(
        description="Contact method type: email_hash, platform_handle, legal_entity"
    )
    contact_hash: str = Field(
        description="SHA-256 of actual contact info (verifiable without revealing)"
    )


class SessionResultResponse(BaseModel):
    """Final results for a completed session."""

    session_id: str
    status: str
    suites_completed: list[str]
    results: dict[str, Any]
    overall_passed: bool
    tier: str | None = Field(default=None, description="METTLE verification tier (bronze/silver/gold/platinum)")
    iteration_curve: dict[str, Any] | None = Field(default=None, description="Only for sessions including Suite 10")
    vcp_attestation: dict[str, Any] | None = Field(
        default=None,
        description="VCP-compatible attestation (when include_vcp=true)",
    )
    governance_attestation: GovernanceAttestation | None = Field(
        default=None,
        description="Governance framework attestation (for Platinum tier)",
    )
    operator_attestation: OperatorAttestation | None = Field(
        default=None,
        description="Operator accountability chain (cryptographic link agent -> operator)",
    )
    elapsed_ms: int


class SuiteInfoResponse(BaseModel):
    """Information about a single verification suite."""

    name: str
    display_name: str
    description: str
    suite_number: int
    is_multi_round: bool
    difficulty_levels: list[str]
    available: bool = True


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
