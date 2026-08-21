"""Tests for METTLE Pydantic models."""

from datetime import datetime, timedelta, timezone

import pytest
from mettle.models import (
    BadgeInfo,
    Challenge,
    ChallengeRequest,
    ChallengeType,
    Difficulty,
    MettleResult,
    MettleSession,
    VerificationResult,
)
from pydantic import ValidationError


def _verification_results(
    passed: tuple[bool, ...] = (True, True, True),
) -> list[VerificationResult]:
    challenge_types = (
        ChallengeType.SPEED_MATH,
        ChallengeType.TOKEN_PREDICTION,
        ChallengeType.INSTRUCTION_FOLLOWING,
    )
    return [
        VerificationResult(
            challenge_id=f"mtl_{index}",
            challenge_type=challenge_types[index],
            passed=did_pass,
            response_time_ms=100,
            time_limit_ms=1000,
        )
        for index, did_pass in enumerate(passed)
    ]


class TestChallengeType:
    """Test ChallengeType enum."""

    def test_all_types_defined(self):
        """Verify all challenge types are defined."""
        assert ChallengeType.SPEED_MATH.value == "speed_math"
        assert ChallengeType.CHAINED_REASONING.value == "chained_reasoning"
        assert ChallengeType.TOKEN_PREDICTION.value == "token_prediction"
        assert ChallengeType.INSTRUCTION_FOLLOWING.value == "instruction_following"
        assert ChallengeType.CONSISTENCY.value == "consistency"

    def test_type_count(self):
        """Verify expected number of challenge types."""
        assert len(ChallengeType) == 5


class TestDifficulty:
    """Test Difficulty enum."""

    def test_basic_and_full(self):
        """Verify difficulty levels."""
        assert Difficulty.BASIC.value == "basic"
        assert Difficulty.FULL.value == "full"

    def test_difficulty_count(self):
        """Verify expected number of difficulties."""
        assert len(Difficulty) == 2


class TestChallenge:
    """Test Challenge model."""

    def test_create_challenge(self):
        """Test creating a valid challenge."""
        challenge = Challenge(
            id="mtl_abc123",
            type=ChallengeType.SPEED_MATH,
            prompt="Calculate: 2 + 2",
            data={"expected_answer": 4},
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            time_limit_ms=5000,
        )
        assert challenge.id == "mtl_abc123"
        assert challenge.type == ChallengeType.SPEED_MATH
        assert challenge.time_limit_ms == 5000

    def test_challenge_auto_issued_at(self):
        """Test that issued_at is auto-populated."""
        challenge = Challenge(
            id="mtl_test",
            type=ChallengeType.SPEED_MATH,
            prompt="Test",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            time_limit_ms=1000,
        )
        assert challenge.issued_at is not None

    def test_challenge_requires_id(self):
        """Test that id is required."""
        with pytest.raises(ValidationError):
            Challenge.model_validate(
                {
                    "type": ChallengeType.SPEED_MATH,
                    "prompt": "Test",
                    "expires_at": datetime.now(timezone.utc) + timedelta(minutes=5),
                    "time_limit_ms": 1000,
                }
            )

    def test_challenge_requires_prompt(self):
        """Test that prompt is required."""
        with pytest.raises(ValidationError):
            Challenge.model_validate(
                {
                    "id": "mtl_test",
                    "type": ChallengeType.SPEED_MATH,
                    "expires_at": datetime.now(timezone.utc) + timedelta(minutes=5),
                    "time_limit_ms": 1000,
                }
            )

    def test_challenge_data_default_empty(self):
        """Test that data defaults to empty dict."""
        challenge = Challenge(
            id="mtl_test",
            type=ChallengeType.SPEED_MATH,
            prompt="Test",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            time_limit_ms=1000,
        )
        assert challenge.data == {}

    @pytest.mark.parametrize("time_limit_ms", [True, 0, -1, 300_001, 1.5, "1000"])
    def test_time_limit_rejects_bool_out_of_range_and_coercion(self, time_limit_ms):
        with pytest.raises(ValidationError):
            Challenge(
                id="mtl_test",
                type=ChallengeType.SPEED_MATH,
                prompt="Test",
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                time_limit_ms=time_limit_ms,
            )

    @pytest.mark.parametrize("time_limit_ms", [1, 300_000])
    def test_time_limit_accepts_exact_boundaries(self, time_limit_ms):
        challenge = Challenge(
            id="mtl_test",
            type=ChallengeType.SPEED_MATH,
            prompt="Test",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            time_limit_ms=time_limit_ms,
        )
        assert challenge.time_limit_ms == time_limit_ms

    @pytest.mark.parametrize(
        "changes",
        [
            {"id": "   "},
            {"prompt": "\t"},
            {"issued_at": datetime.now()},
            {"expires_at": datetime.now() + timedelta(minutes=5)},
        ],
    )
    def test_rejects_blank_text_and_naive_timestamps(self, changes):
        values = {
            "id": "mtl_test",
            "type": ChallengeType.SPEED_MATH,
            "prompt": "Test",
            "issued_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc) + timedelta(minutes=5),
            "time_limit_ms": 1000,
        }
        values.update(changes)
        with pytest.raises(ValidationError):
            Challenge.model_validate(values)

    def test_expiry_must_follow_issue_time(self):
        now = datetime.now(timezone.utc)
        with pytest.raises(ValidationError, match="later than issued_at"):
            Challenge(
                id="mtl_test",
                type=ChallengeType.SPEED_MATH,
                prompt="Test",
                issued_at=now,
                expires_at=now,
                time_limit_ms=1000,
            )


class TestChallengeRequest:
    """Test ChallengeRequest model."""

    def test_default_difficulty(self):
        """Test default difficulty is BASIC."""
        request = ChallengeRequest.model_validate({})
        assert request.difficulty == Difficulty.BASIC

    def test_optional_entity_id(self):
        """Test entity_id is optional."""
        request = ChallengeRequest.model_validate({})
        assert request.entity_id is None

        request_with_id = ChallengeRequest(entity_id="agent-001")
        assert request_with_id.entity_id == "agent-001"


class TestVerificationResult:
    """Test VerificationResult model."""

    def test_create_result(self):
        """Test creating a verification result."""
        result = VerificationResult(
            challenge_id="mtl_123",
            challenge_type=ChallengeType.SPEED_MATH,
            passed=True,
            response_time_ms=1234,
            time_limit_ms=5000,
        )
        assert result.passed
        assert result.response_time_ms == 1234

    def test_result_details_default_empty(self):
        """Test that details defaults to empty dict."""
        result = VerificationResult(
            challenge_id="mtl_123",
            challenge_type=ChallengeType.SPEED_MATH,
            passed=True,
            response_time_ms=1000,
            time_limit_ms=5000,
        )
        assert result.details == {}

    @pytest.mark.parametrize(
        "changes",
        [
            {"passed": "false"},
            {"response_time_ms": True},
            {"response_time_ms": -1},
            {"time_limit_ms": True},
            {"time_limit_ms": 0},
            {"time_limit_ms": 300_001},
        ],
    )
    def test_rejects_coerced_booleans_and_invalid_numeric_bounds(self, changes):
        values = {
            "challenge_id": "mtl_123",
            "challenge_type": ChallengeType.SPEED_MATH,
            "passed": True,
            "response_time_ms": 0,
            "time_limit_ms": 300_000,
        }
        values.update(changes)
        with pytest.raises(ValidationError):
            VerificationResult.model_validate(values)


class TestMettleResult:
    """Test MettleResult model."""

    def test_create_mettle_result(self):
        """Test creating a METTLE result."""
        results = _verification_results()
        result = MettleResult.model_validate(
            {
                "entity_id": "agent-001",
                "verified": True,
                "screening_passed": True,
                "credential_eligible": True,
                "passed": 3,
                "total": 3,
                "pass_rate": 1.0,
                "results": results,
            }
        )
        assert result.verified
        assert result.pass_rate == 1.0
        assert result.badge is None

    def test_mettle_result_auto_issued_at(self):
        """Test that issued_at is auto-populated."""
        result = MettleResult.model_validate(
            {
                "entity_id": None,
                "verified": False,
                "passed": 0,
                "total": 0,
                "pass_rate": 0.0,
                "results": [],
            }
        )
        assert result.issued_at is not None

    @pytest.mark.parametrize(
        "changes",
        [
            {"passed": True},
            {"total": True},
            {"pass_rate": True},
            {"pass_rate": "1.0"},
            {"pass_rate": float("nan")},
            {"pass_rate": float("inf")},
            {"pass_rate": -0.1},
            {"pass_rate": 1.1},
        ],
    )
    def test_rejects_bool_coercion_and_invalid_summary_numbers(self, changes):
        values = {
            "entity_id": None,
            "verified": True,
            "screening_passed": True,
            "credential_eligible": True,
            "passed": 3,
            "total": 3,
            "pass_rate": 1.0,
            "results": _verification_results(),
        }
        values.update(changes)
        with pytest.raises(ValidationError):
            MettleResult.model_validate(values)

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"passed": 2}, "number of passing results"),
            ({"total": 4}, "number of results"),
            ({"pass_rate": 0.9}, "passed / total"),
            ({"verified": False}, "verification flags"),
        ],
    )
    def test_rejects_internally_inconsistent_summary(self, changes, message):
        values = {
            "entity_id": None,
            "verified": True,
            "screening_passed": True,
            "credential_eligible": True,
            "passed": 3,
            "total": 3,
            "pass_rate": 1.0,
            "results": _verification_results(),
        }
        values.update(changes)
        with pytest.raises(ValidationError, match=message):
            MettleResult.model_validate(values)


class TestBadgeInfo:
    def test_real_generated_badge_exceeds_identifier_limit_but_is_accepted(self):
        from main import generate_signed_badge

        generated = generate_signed_badge(
            entity_id="x" * 256,
            difficulty=Difficulty.BASIC.value,
            pass_rate=1.0,
            session_id="ses_model_badge",
        )
        assert len(generated["token"]) > 256

        badge = BadgeInfo(
            token=generated["token"],
            expires_at=datetime.fromisoformat(generated["expires_at"]),
            freshness_nonce=generated["freshness_nonce"],
            signed=generated["signed"],
            jti=generated["jti"],
        )
        assert badge.token == generated["token"]

    def test_token_has_a_bounded_8192_character_domain(self):
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
        badge = BadgeInfo.model_validate(
            {"token": "x" * 8192, "expires_at": expires_at}
        )
        assert len(badge.token) == 8192
        with pytest.raises(ValidationError):
            BadgeInfo.model_validate({"token": "x" * 8193, "expires_at": expires_at})

    def test_signed_flag_is_strict(self):
        with pytest.raises(ValidationError):
            BadgeInfo.model_validate(
                {
                    "token": "signed-token",
                    "expires_at": datetime.now(timezone.utc) + timedelta(minutes=5),
                    "signed": "false",
                }
            )


class TestMettleSession:
    """Test MettleSession model."""

    def test_create_session(self):
        """Test creating a session."""
        session = MettleSession(
            session_id="ses_abc123",
            entity_id="agent-001",
            difficulty=Difficulty.BASIC,
            challenges=[],
            access_token_hash="hash",
        )
        assert session.session_id == "ses_abc123"
        assert session.difficulty == Difficulty.BASIC
        assert not session.completed

    def test_session_defaults(self):
        """Test session default values."""
        session = MettleSession(
            session_id="ses_test",
            entity_id=None,
            difficulty=Difficulty.BASIC,
            challenges=[],
            access_token_hash="hash",
        )
        assert session.results == []
        assert session.started_at is not None
        assert not session.completed

    def test_completed_flag_is_strict(self):
        with pytest.raises(ValidationError):
            MettleSession.model_validate(
                {
                    "session_id": "ses_test",
                    "entity_id": None,
                    "difficulty": Difficulty.BASIC,
                    "challenges": [],
                    "completed": "false",
                    "access_token_hash": "hash",
                }
            )

    def test_partial_basic_battery_is_rejected_as_corrupt_state(self):
        challenge = Challenge(
            id="mtl_only",
            type=ChallengeType.SPEED_MATH,
            prompt="Calculate: 1 + 1",
            data={"expected_answer": 2},
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            time_limit_ms=1000,
        )
        with pytest.raises(ValidationError, match="selected difficulty"):
            MettleSession(
                session_id="ses_partial",
                entity_id=None,
                difficulty=Difficulty.BASIC,
                challenges=[challenge],
                access_token_hash="hash",
            )

    def test_badge_requires_a_completed_session(self):
        badge = BadgeInfo.model_validate(
            {
                "token": "signed-token",
                "expires_at": datetime.now(timezone.utc) + timedelta(minutes=5),
                "signed": True,
            }
        )
        with pytest.raises(ValidationError, match="badge_info requires"):
            MettleSession(
                session_id="ses_incomplete_badge",
                entity_id=None,
                difficulty=Difficulty.BASIC,
                challenges=[],
                access_token_hash="hash",
                badge_info=badge,
            )
