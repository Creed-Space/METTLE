"""METTLE: Pydantic models for challenge/response protocol."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ChallengeType(str, Enum):
    SPEED_MATH = "speed_math"
    CHAINED_REASONING = "chained_reasoning"
    # B105 is a false positive here; this is a public challenge type identifier.
    TOKEN_PREDICTION = "token_prediction"  # nosec B105
    INSTRUCTION_FOLLOWING = "instruction_following"
    CONSISTENCY = "consistency"


class Difficulty(str, Enum):
    BASIC = "basic"
    FULL = "full"


class Challenge(BaseModel):
    """A METTLE challenge to be solved."""

    id: str = Field(..., description="Unique challenge ID")
    type: ChallengeType
    prompt: str = Field(..., description="The challenge prompt/question")
    data: dict[str, Any] = Field(
        default_factory=dict, description="Additional challenge data"
    )
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime
    time_limit_ms: int = Field(..., description="Maximum allowed response time in ms")

    def sanitized(self) -> "Challenge":
        """Return a copy with sensitive data (answers) removed for client response."""
        # Keys that contain answers - never expose to client
        secret_keys = {"expected_answer", "chain", "instructions"}
        clean_data = {k: v for k, v in self.data.items() if k not in secret_keys}
        return Challenge(
            id=self.id,
            type=self.type,
            prompt=self.prompt,
            data=clean_data,
            issued_at=self.issued_at,
            expires_at=self.expires_at,
            time_limit_ms=self.time_limit_ms,
        )


class ChallengeRequest(BaseModel):
    """Request for a new challenge."""

    difficulty: Difficulty = Difficulty.BASIC
    entity_id: str | None = Field(None, description="Optional entity identifier")


class ChallengeResponse(BaseModel):
    """Response to a challenge."""

    challenge_id: str
    answer: Any
    entity_id: str | None = None


class VerificationResult(BaseModel):
    """Result of verifying a challenge response."""

    challenge_id: str
    challenge_type: ChallengeType
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)
    response_time_ms: int
    time_limit_ms: int


class BadgeInfo(BaseModel):
    """Server-issued METTLE badge metadata."""

    token: str = Field(..., description="The badge token (JWT or simple)")
    expires_at: datetime = Field(..., description="When the badge expires")
    freshness_nonce: str | None = Field(
        None, description="Nonce for freshness verification"
    )
    signed: bool = Field(
        False, description="Whether the badge is cryptographically signed"
    )
    jti: str | None = Field(None, description="Unique badge ID for revocation")


class MettleResult(BaseModel):
    """Overall METTLE reverse-CAPTCHA verification result."""

    entity_id: str | None
    verified: bool = Field(
        default=False,
        description="Whether the configured METTLE challenge threshold was met",
    )
    screening_passed: bool = Field(
        default=False,
        description="Compatibility alias for the challenge-threshold outcome",
    )
    assurance: str = Field(default="mettle_behavioral_verification")
    credential_eligible: bool = Field(default=False)
    tier: str = Field(default="none", description="METTLE tier earned by this session")
    passed: int
    total: int
    pass_rate: float
    results: list[VerificationResult]
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    badge: str | None = Field(
        None, description="Signed, time-limited METTLE badge when one was issued"
    )
    badge_info: BadgeInfo | None = Field(
        None, description="Metadata for the server-issued badge"
    )


class MettleSession(BaseModel):
    """A METTLE verification session."""

    session_id: str
    entity_id: str | None
    difficulty: Difficulty
    challenges: list[Challenge]
    results: list[VerificationResult] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed: bool = False
    access_token_hash: str = Field(
        description="SHA-256 digest of the bearer token for legacy session access"
    )
    badge_info: BadgeInfo | None = Field(
        default=None,
        description="Stable server-issued badge metadata for a completed session",
    )
