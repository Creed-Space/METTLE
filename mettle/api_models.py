"""Pydantic models for METTLE API.

Request/response models for session management, verification, and multi-round challenges.
"""

from __future__ import annotations

import enum
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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

MAX_SUITES_PER_SESSION = len(SUITE_NAMES)
MAX_ANSWER_BYTES = 64 * 1024
MAX_ATTESTATION_BYTES = 128 * 1024


def _validate_answer_object(value: dict[str, Any]) -> dict[str, Any]:
    """Bound evaluator input before iteration, persistence, or LLM use."""
    if len(value) > 100:
        raise ValueError("Answer object contains too many top-level fields")
    try:
        encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("Answers must be JSON serializable") from exc
    if len(encoded) > MAX_ANSWER_BYTES:
        raise ValueError(f"Answer payload exceeds {MAX_ANSWER_BYTES} bytes")
    return value


# ---- Request Models ----


class PresenceRegistration(BaseModel):
    """Opt-in key binding for a METTLE Presence Protocol session."""

    public_key_pem: str = Field(
        min_length=1,
        max_length=8192,
        description="Ed25519 public key used for session and presentation proofs",
    )
    audience: str = Field(
        min_length=1,
        max_length=256,
        description="Intended verifier or service audience for the credential",
    )

    @field_validator("audience")
    @classmethod
    def validate_audience(_cls, value: str) -> str:
        if value != value.strip() or any(ord(char) < 0x21 for char in value):
            raise ValueError("audience must be a trimmed printable identifier")
        return value


class PresenceProof(BaseModel):
    """Holder signature over one server-issued session submission message."""

    nonce: str = Field(min_length=32, max_length=256)
    previous_transcript_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    signature: str = Field(
        min_length=80,
        max_length=128,
        description="Base64 Ed25519 signature",
    )


class PresenceStateReceipt(BaseModel):
    """Issuer signature authenticating one exact public Presence state."""

    key_id: str = Field(min_length=1, max_length=256)
    algorithm: Literal["Ed25519"] = "Ed25519"
    signature: str = Field(min_length=80, max_length=128)


class PresenceState(BaseModel):
    """Client-safe state required to sign the next Presence submission."""

    protocol: Literal["mettle-presence-v1"] = "mettle-presence-v1"
    key_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    audience: str
    nonce: str | None
    transcript_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    sequence: int = Field(ge=0)
    action: str | None = None
    completed: bool = False
    continuity_protocol: Literal["mettle-continuity-v1"] | None = None
    issuer_receipt: PresenceStateReceipt


class CreateSessionRequest(BaseModel):
    """Request to start a METTLE verification session."""

    # Unknown request fields are rejected rather than silently discarded. In
    # particular, this prevents the retired one-step operator commitment field
    # from looking accepted even though a replay-safe protocol would require a
    # server-issued nonce before the operator signs.
    model_config = ConfigDict(extra="forbid")

    suites: list[str] = Field(
        default=["all"],
        min_length=1,
        max_length=MAX_SUITES_PER_SESSION,
        description="Suite names or 'all'",
    )
    difficulty: Literal["easy", "standard", "hard"] = "standard"
    entity_id: str | None = Field(
        default=None,
        max_length=256,
        description="Optional entity identifier",
    )
    vcp_token: str | None = Field(
        default=None,
        max_length=32768,
        description="Optional CSM-1 VCP token for enhanced Suite 9 verification",
    )
    presence: PresenceRegistration | None = Field(
        default=None,
        description="Opt into key-bound session submissions and credential presentation",
    )
    allow_third_party_llm: bool = Field(
        default=False,
        description=(
            "Explicitly acknowledge that llm-dynamic candidate responses are sent "
            "to Anthropic for evaluation during this session"
        ),
    )

    @field_validator("suites")
    @classmethod
    def validate_suites(_cls, suites: list[str]) -> list[str]:
        if len(suites) != len(set(suites)):
            raise ValueError("Duplicate suites are not allowed")
        if "all" in suites and len(suites) != 1:
            raise ValueError("'all' cannot be combined with explicit suites")
        return suites


class RoundAnswerRequest(BaseModel):
    """Submit answers for a multi-round challenge round."""

    answers: dict[str, Any] = Field(description="Challenge-specific answers")
    submitted_at: datetime | None = Field(
        default=None, description="Client-side timestamp"
    )
    presence_proof: PresenceProof | None = None

    _bound_answers = field_validator("answers")(_validate_answer_object)


class VerifyRequest(BaseModel):
    """Submit answers for a single-shot suite."""

    suite: str = Field(description="Suite name to verify")
    answers: dict[str, Any] = Field(description="Suite-specific answers")
    presence_proof: PresenceProof | None = None

    _bound_answers = field_validator("answers")(_validate_answer_object)


# ---- Response Models ----


class CreateSessionResponse(BaseModel):
    """Response after creating a verification session."""

    session_id: str
    created_at: datetime
    expires_at: datetime
    suites: list[str]
    challenges: dict[str, Any] = Field(
        description="Suite name -> challenge data (no answers)"
    )
    time_budget_ms: int
    presence: PresenceState | None = None


class RoundFeedbackResponse(BaseModel):
    """Feedback after a multi-round answer submission."""

    round_num: int
    accuracy: float
    errors: list[str]
    feedback: dict[str, Any]
    time_remaining_ms: int
    next_round_data: dict[str, Any] | None = Field(
        default=None, description="Data for next round; null if final"
    )
    presence: PresenceState | None = None
    next_challenge: dict[str, Any] | None = None


class VerifyResponse(BaseModel):
    """Result of a single-shot suite verification."""

    suite: str
    passed: bool
    score: float
    details: dict[str, Any]
    presence: PresenceState | None = None
    next_challenge: dict[str, Any] | None = None


class PresentationChallengeRequest(BaseModel):
    """Request a fresh, single-use proof-of-possession challenge."""

    credential_jti: str = Field(pattern=r"^[0-9a-f]{32}$")
    audience: str = Field(min_length=1, max_length=256)

    @field_validator("audience")
    @classmethod
    def validate_audience(_cls, value: str) -> str:
        return PresenceRegistration.validate_audience(value)


class CredentialStatusRequest(BaseModel):
    """Request a fresh issuer-authenticated revocation status receipt."""

    model_config = ConfigDict(extra="forbid")
    credential_jti: str = Field(pattern=r"^[0-9a-f]{32}$")


class CredentialStatusResponse(BaseModel):
    """Short-lived signed status for portable credential acceptance."""

    protocol: Literal["mettle-credential-status-v1"]
    auditor: Literal["mettle.creed.space"]
    auditor_key_id: str
    credential_jti: str
    status: Literal["good", "revoked"]
    # Keep the exact signed RFC 3339 strings. Parsing and reserializing a
    # datetime may normalize ``+00:00`` to ``Z`` and invalidate the signature.
    observed_at: str = Field(min_length=20, max_length=64)
    expires_at: str = Field(min_length=20, max_length=64)
    signature: str


class PresentationChallengeResponse(BaseModel):
    """Fresh challenge to be signed by the credential's bound holder key."""

    challenge_id: str
    nonce: str
    audience: str
    credential_jti: str
    expires_at: datetime


class PresentationVerifyRequest(BaseModel):
    """Verify one issuer-signed credential and live holder signature."""

    challenge_id: str = Field(min_length=32, max_length=256)
    attestation: dict[str, Any]
    holder_signature: str = Field(min_length=80, max_length=128)

    @field_validator("attestation")
    @classmethod
    def validate_attestation(_cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 32:
            raise ValueError("Attestation contains too many top-level fields")
        try:
            encoded = json.dumps(value, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("Attestation must be JSON serializable") from exc
        if len(encoded) > MAX_ATTESTATION_BYTES:
            raise ValueError(f"Attestation exceeds {MAX_ATTESTATION_BYTES} bytes")
        return value


class PresentationVerifyResponse(BaseModel):
    """Successful live verification of a key-bound METTLE credential."""

    valid: Literal[True] = True
    credential_jti: str
    audience: str
    tier: str
    subject_id: str
    entity_id: str | None = None
    key_fingerprint: str
    transcript_hash: str


class GovernanceAttestation(BaseModel):
    """Attests the governance framework governing an agent.

    Populated during METTLE verification when the agent provides a VCP token
    containing Creed governance metadata. Enables platforms to distinguish
    between governed and ungoverned agents.

    Parsed governance metadata never increases a METTLE tier. Tiers are earned
    only by passing the configured challenge suite ranges.
    """

    entity_id: str | None = Field(
        default=None, description="Entity claim associated with the source VCP token"
    )
    session_id: str = Field(description="METTLE session this attestation belongs to")
    tier: str = Field(description="METTLE tier at attestation construction time")
    source_vcp_hash: str = Field(description="SHA-256 hash of the supplied VCP token")
    source_verified: bool = Field(
        default=False,
        description="Whether the source VCP provenance was cryptographically verified",
    )

    framework: str = Field(
        description="Governance framework: creed-space, custom, none"
    )
    framework_version: str | None = Field(
        default=None, description="Framework version (e.g. 2.1.0)"
    )
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
    observed_at: datetime = Field(description="When the unverified metadata was parsed")
    expires_at: datetime = Field(description="When this metadata snapshot expires")
    attestation_signature: str | None = Field(
        default=None,
        description="Reserved for a future externally verified provenance flow",
    )


class SessionResultResponse(BaseModel):
    """Final results for a completed session."""

    session_id: str
    status: str
    suites_completed: list[str]
    results: dict[str, Any]
    overall_passed: bool
    verified: bool = Field(
        default=False,
        description="Whether every selected challenge suite passed",
    )
    assurance: str = Field(
        default="mettle_behavioral_verification",
        description="Class of METTLE verification represented by this result",
    )
    credential_eligible: bool = Field(
        default=False,
        description="Whether a complete tier-qualifying suite range passed",
    )
    credential_suites_passed: list[str] = Field(
        default_factory=list,
        description="Passed suites backed by tier-eligible server evidence",
    )
    supplemental_suites_passed: list[str] = Field(
        default_factory=list,
        description="Passed observational suites that cannot raise a tier",
    )
    tier: str = Field(
        default="none",
        description="Highest contiguous METTLE challenge tier earned",
    )
    iteration_curve: dict[str, Any] | None = Field(
        default=None, description="Only for sessions including Suite 10"
    )
    vcp_attestation: dict[str, Any] | None = Field(
        default=None,
        description="VCP-compatible result or signed credential when requested",
    )
    governance_attestation: GovernanceAttestation | None = Field(
        default=None,
        description="Unverified governance metadata parsed from the supplied VCP token",
    )
    presence: PresenceState | None = Field(
        default=None,
        description="Final key-bound session transcript state, when requested at creation",
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
    presence: PresenceState | None = None
    elapsed_ms: int
