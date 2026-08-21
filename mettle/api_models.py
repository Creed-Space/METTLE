"""Pydantic models for METTLE API.

Request/response models for session management, verification, and multi-round challenges.
"""

from __future__ import annotations

import enum
import base64
import json
import math
from datetime import datetime, timedelta
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
MAX_JSON_DEPTH = 16
MAX_JSON_NODES = 4096


class StrictRequestModel(BaseModel):
    """Base class for public request bodies.

    Silently ignored fields turn client mistakes into ambiguous requests and can
    hide unsupported security-relevant claims.  Request schemas therefore reject
    unknown fields at every nested model boundary.
    """

    model_config = ConfigDict(extra="forbid")


def validate_bounded_json(
    value: Any,
    *,
    max_bytes: int,
    max_depth: int = MAX_JSON_DEPTH,
    max_nodes: int = MAX_JSON_NODES,
    label: str = "JSON value",
) -> Any:
    """Validate a JSON-compatible value without unbounded traversal.

    The iterative walk rejects cycles and repeated container aliases, excessive
    depth or node counts, non-string object keys, unsupported Python objects, and
    non-finite floats before serialisation.  A cheap UTF-8 estimate prevents a
    single huge scalar from forcing a large temporary allocation; the final
    canonical serialisation enforces the exact byte limit.
    """

    if max_bytes < 1 or max_depth < 0 or max_nodes < 1:
        raise ValueError("JSON validation limits must be positive")

    stack: list[tuple[Any, int]] = [(value, 0)]
    seen_containers: set[int] = set()
    nodes = 0
    estimated_bytes = 0

    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise ValueError(f"{label} contains too many values")
        if depth > max_depth:
            raise ValueError(f"{label} exceeds maximum nesting depth")

        if item is None:
            estimated_bytes += 4
        elif isinstance(item, bool):
            estimated_bytes += 4 if item else 5
        elif isinstance(item, int):
            estimated_bytes += len(str(item))
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{label} contains a non-finite number")
            estimated_bytes += len(json.dumps(item, allow_nan=False))
        elif isinstance(item, str):
            # Every Unicode code point needs at least one UTF-8 byte. Reject an
            # obviously oversized scalar before allocating its encoded copy;
            # only values within a small bounded multiplier reach encode().
            if estimated_bytes + len(item) + 2 > max_bytes:
                raise ValueError(f"{label} exceeds {max_bytes} bytes")
            estimated_bytes += len(item.encode("utf-8")) + 2
        elif isinstance(item, dict):
            identity = id(item)
            if identity in seen_containers:
                raise ValueError(f"{label} contains a cycle or repeated container")
            seen_containers.add(identity)
            estimated_bytes += 2
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError(f"{label} object keys must be strings")
                if estimated_bytes + len(key) + 3 > max_bytes:
                    raise ValueError(f"{label} exceeds {max_bytes} bytes")
                estimated_bytes += len(key.encode("utf-8")) + 3
                stack.append((child, depth + 1))
        elif isinstance(item, list):
            identity = id(item)
            if identity in seen_containers:
                raise ValueError(f"{label} contains a cycle or repeated container")
            seen_containers.add(identity)
            estimated_bytes += 2
            stack.extend((child, depth + 1) for child in item)
        else:
            raise ValueError(f"{label} must contain only JSON-compatible values")

        if estimated_bytes > max_bytes:
            raise ValueError(f"{label} exceeds {max_bytes} bytes")

    try:
        encoded = json.dumps(
            value,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} must be JSON serializable") from exc
    if len(encoded) > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes} bytes")
    return value


def _validate_answer_object(value: dict[str, Any]) -> dict[str, Any]:
    """Bound evaluator input before iteration, persistence, or LLM use."""
    if len(value) > 100:
        raise ValueError("Answer object contains too many top-level fields")
    try:
        validate_bounded_json(
            value,
            max_bytes=MAX_ANSWER_BYTES,
            label="Answer payload",
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    return value


# ---- Request Models ----


class OperatorCommitment(StrictRequestModel):
    """Operator accountability commitment submitted with session creation.

    The operator signs a commitment accepting accountability for the agent.
    Ed25519 signature is verified server-side before attestation is issued.
    """

    operator_pseudonym: str = Field(
        min_length=1,
        max_length=256,
        description="Operator identifier (can be pseudonymous)",
    )
    operator_public_key: str = Field(
        min_length=1,
        max_length=8192,
        description="Ed25519 public key (PEM format)",
    )
    signed_commitment: str = Field(
        min_length=88,
        max_length=88,
        description="Base64 Ed25519 signature over the canonical version-1 operator commitment JSON",
    )
    contact_method: Literal["email_hash", "platform_handle", "legal_entity"] = Field(
        description="Contact method type: email_hash, platform_handle, legal_entity",
    )
    contact_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 of actual contact info (verifiable without revealing)",
    )
    issued_at: datetime = Field(
        description="UTC time at which the operator signed this commitment",
    )
    nonce: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Single-use 32-byte cryptographic nonce encoded as lowercase hex",
    )

    @field_validator("operator_pseudonym")
    @classmethod
    def validate_operator_pseudonym(_cls, value: str) -> str:
        if value != value.strip() or not value.isprintable():
            raise ValueError("operator_pseudonym must be a trimmed printable string")
        return value

    @field_validator("signed_commitment")
    @classmethod
    def validate_signature(_cls, value: str) -> str:
        try:
            decoded = base64.b64decode(value, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("signed_commitment must be canonical base64") from exc
        if len(decoded) != 64:
            raise ValueError("signed_commitment must contain an Ed25519 signature")
        return value

    @field_validator("issued_at")
    @classmethod
    def validate_issued_at(_cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("issued_at must include a UTC timezone")
        if value.utcoffset() != timedelta(0):
            raise ValueError("issued_at must use UTC")
        return value


class PresenceRegistration(StrictRequestModel):
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


class PresenceProof(StrictRequestModel):
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


class CreateSessionRequest(StrictRequestModel):
    """Request to start a METTLE verification session."""

    suites: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(
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
    operator_commitment: OperatorCommitment | None = Field(
        default=None,
        description="Optional signed operator statement; does not create a trust tier",
    )
    presence: PresenceRegistration | None = Field(
        default=None,
        description="Opt into key-bound session submissions and credential presentation",
    )

    @field_validator("suites")
    @classmethod
    def validate_suites(_cls, suites: list[str]) -> list[str]:
        if len(suites) != len(set(suites)):
            raise ValueError("Duplicate suites are not allowed")
        if "all" in suites and len(suites) != 1:
            raise ValueError("'all' cannot be combined with explicit suites")
        return suites

    @model_validator(mode="after")
    def validate_operator_subject(self) -> "CreateSessionRequest":
        if self.operator_commitment is not None and not self.entity_id:
            raise ValueError("entity_id is required with an operator commitment")
        return self


class RoundAnswerRequest(StrictRequestModel):
    """Submit answers for a multi-round challenge round."""

    answers: dict[str, Any] = Field(description="Challenge-specific answers")
    submitted_at: datetime | None = Field(
        default=None, description="Client-side timestamp"
    )
    presence_proof: PresenceProof | None = None

    _bound_answers = field_validator("answers")(_validate_answer_object)


class VerifyRequest(StrictRequestModel):
    """Submit answers for a single-shot suite."""

    suite: str = Field(min_length=1, max_length=64, description="Suite name to verify")
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


class PresentationChallengeRequest(StrictRequestModel):
    """Request a fresh, single-use proof-of-possession challenge."""

    credential_jti: str = Field(pattern=r"^[0-9a-f]{32}$")
    audience: str = Field(min_length=1, max_length=256)

    @field_validator("audience")
    @classmethod
    def validate_audience(_cls, value: str) -> str:
        return PresenceRegistration.validate_audience(value)


class PresentationChallengeResponse(BaseModel):
    """Fresh challenge to be signed by the credential's bound holder key."""

    challenge_id: str
    nonce: str
    audience: str
    credential_jti: str
    expires_at: datetime


class PresentationVerifyRequest(StrictRequestModel):
    """Verify one issuer-signed credential and live holder signature."""

    challenge_id: str = Field(min_length=32, max_length=256)
    attestation: dict[str, Any]
    holder_signature: str = Field(min_length=80, max_length=128)

    @field_validator("attestation")
    @classmethod
    def validate_attestation(_cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 32:
            raise ValueError("Attestation contains too many top-level fields")
        validate_bounded_json(
            value,
            max_bytes=MAX_ATTESTATION_BYTES,
            max_nodes=MAX_JSON_NODES * 2,
            label="Attestation",
        )
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


class OperatorAttestation(BaseModel):
    """Cryptographic link from agent to operator.

    Even pseudonymous operators provide a verifiable accountability chain.
    The contact_hash allows platforms to verify contact info exists without
    revealing it publicly. If the agent causes harm, the platform can request
    the operator reveal themselves by providing the preimage.
    """

    operator_pseudonym: str = Field(
        min_length=1,
        max_length=256,
        description="Operator identifier (can be pseudonymous)",
    )
    operator_public_key: str = Field(
        min_length=1,
        max_length=8192,
        description="Ed25519 public key (PEM format)",
    )
    operator_signed_commitment: str = Field(
        min_length=88,
        max_length=88,
        description="Operator signs: 'I accept accountability for agent {entity_id}'",
    )
    commitment_timestamp: datetime = Field(
        description="Signed UTC time at which the operator made the commitment"
    )
    commitment_nonce: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Signed single-use commitment nonce",
    )
    contact_method: Literal["email_hash", "platform_handle", "legal_entity"] = Field(
        description="Contact method type: email_hash, platform_handle, legal_entity"
    )
    contact_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 of actual contact info (verifiable without revealing)",
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
    operator_attestation: OperatorAttestation | None = Field(
        default=None,
        description="Operator accountability chain (cryptographic link agent -> operator)",
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
