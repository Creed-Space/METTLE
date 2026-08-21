"""METTLE: Pydantic models for challenge/response protocol."""

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


Identifier = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)
]
Prompt = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=16_384)
]
BadgeToken = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8192)
]


def _require_aware(value: datetime, field_name: str) -> datetime:
    """Reject ambiguous naive timestamps at protocol boundaries."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value


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

    id: Identifier = Field(..., description="Unique challenge ID")
    type: ChallengeType
    prompt: Prompt = Field(..., description="The challenge prompt/question")
    data: dict[str, Any] = Field(
        default_factory=dict, description="Additional challenge data"
    )
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime
    time_limit_ms: int = Field(
        ..., strict=True, gt=0, le=300_000, description="Maximum response time in ms"
    )

    @field_validator("issued_at", "expires_at")
    @classmethod
    def timestamps_are_aware(cls, value: datetime, info: Any) -> datetime:
        return _require_aware(value, info.field_name)

    @model_validator(mode="after")
    def expiry_follows_issue(self) -> "Challenge":
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be later than issued_at")
        return self

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
    entity_id: Identifier | None = Field(None, description="Optional entity identifier")


class ChallengeResponse(BaseModel):
    """Response to a challenge."""

    challenge_id: Identifier
    answer: Any
    entity_id: Identifier | None = None


class VerificationResult(BaseModel):
    """Result of verifying a challenge response."""

    challenge_id: Identifier
    challenge_type: ChallengeType
    passed: bool = Field(strict=True)
    details: dict[str, Any] = Field(default_factory=dict)
    response_time_ms: int = Field(strict=True, ge=0)
    time_limit_ms: int = Field(strict=True, gt=0, le=300_000)


class BadgeInfo(BaseModel):
    """Server-issued METTLE badge metadata."""

    token: BadgeToken = Field(..., description="The badge token (JWT or simple)")
    expires_at: datetime = Field(..., description="When the badge expires")
    freshness_nonce: Identifier | None = Field(
        None, description="Nonce for freshness verification"
    )
    signed: bool = Field(
        False, strict=True, description="Whether the badge is cryptographically signed"
    )
    jti: Identifier | None = Field(None, description="Unique badge ID for revocation")

    @field_validator("expires_at")
    @classmethod
    def expiry_is_aware(cls, value: datetime) -> datetime:
        return _require_aware(value, "expires_at")


class MettleResult(BaseModel):
    """Overall METTLE reverse-CAPTCHA verification result."""

    entity_id: Identifier | None
    verified: bool = Field(
        default=False,
        strict=True,
        description="Whether the configured METTLE challenge threshold was met",
    )
    screening_passed: bool = Field(
        default=False,
        strict=True,
        description="Compatibility alias for the challenge-threshold outcome",
    )
    assurance: str = Field(default="mettle_behavioral_verification")
    credential_eligible: bool = Field(default=False, strict=True)
    tier: str = Field(default="none", description="METTLE tier earned by this session")
    passed: int = Field(strict=True, ge=0)
    total: int = Field(strict=True, ge=0)
    pass_rate: float = Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)
    results: list[VerificationResult]
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    badge: BadgeToken | None = Field(
        None, description="Signed, time-limited METTLE badge when one was issued"
    )
    badge_info: BadgeInfo | None = Field(
        None, description="Metadata for the server-issued badge"
    )

    @field_validator("issued_at")
    @classmethod
    def issue_time_is_aware(cls, value: datetime) -> datetime:
        return _require_aware(value, "issued_at")

    @model_validator(mode="after")
    def result_fields_agree(self) -> "MettleResult":
        if self.total != len(self.results):
            raise ValueError("total must equal the number of results")
        actual_passed = sum(result.passed for result in self.results)
        if self.passed != actual_passed:
            raise ValueError("passed must equal the number of passing results")
        expected_rate = self.passed / self.total if self.total else 0.0
        if abs(self.pass_rate - expected_rate) > 1e-9:
            raise ValueError("pass_rate must equal passed / total")
        result_ids = [result.challenge_id for result in self.results]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("results must not contain duplicate challenge IDs")
        expected_verified = self.total >= 3 and self.pass_rate >= 0.8
        if (
            self.verified != expected_verified
            or self.screening_passed != expected_verified
        ):
            raise ValueError("verification flags must match the challenge threshold")
        if self.credential_eligible and not expected_verified:
            raise ValueError("credential eligibility requires successful verification")
        return self


class MettleSession(BaseModel):
    """A METTLE verification session."""

    session_id: Identifier
    entity_id: Identifier | None
    difficulty: Difficulty
    challenges: list[Challenge]
    results: list[VerificationResult] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed: bool = Field(default=False, strict=True)
    access_token_hash: Identifier = Field(
        description="SHA-256 digest of the bearer token for legacy session access"
    )
    badge_info: BadgeInfo | None = Field(
        default=None,
        description="Stable server-issued badge metadata for a completed session",
    )

    @field_validator("started_at")
    @classmethod
    def start_time_is_aware(cls, value: datetime) -> datetime:
        return _require_aware(value, "started_at")

    @model_validator(mode="after")
    def session_state_is_coherent(self) -> "MettleSession":
        challenge_ids = [challenge.id for challenge in self.challenges]
        if len(challenge_ids) != len(set(challenge_ids)):
            raise ValueError("challenges must not contain duplicate IDs")

        # Empty in-progress sessions remain usable as persistence placeholders.
        # Any issued session must match the challenge battery actually generated.
        if self.challenges:
            expected_types = (
                [
                    ChallengeType.SPEED_MATH,
                    ChallengeType.TOKEN_PREDICTION,
                    ChallengeType.INSTRUCTION_FOLLOWING,
                ]
                if self.difficulty == Difficulty.BASIC
                else list(ChallengeType)
            )
            if [challenge.type for challenge in self.challenges] != expected_types:
                raise ValueError("challenges do not match the selected difficulty")
        elif self.results or self.completed:
            raise ValueError(
                "a completed or answered session requires issued challenges"
            )

        if len(self.results) > len(self.challenges):
            raise ValueError("results cannot outnumber challenges")
        for result, challenge in zip(self.results, self.challenges):
            if (
                result.challenge_id != challenge.id
                or result.challenge_type != challenge.type
            ):
                raise ValueError("results must match issued challenges in order")
        if self.completed and len(self.results) != len(self.challenges):
            raise ValueError("completed sessions require one result per challenge")
        if (
            not self.completed
            and self.challenges
            and len(self.results) == len(self.challenges)
        ):
            raise ValueError("fully answered sessions must be marked completed")
        if self.badge_info is not None and not self.completed:
            raise ValueError("badge_info requires a completed session")
        return self
