"""Tests for METTLE Pydantic models."""

from datetime import datetime, timedelta, timezone

import pytest
from mettle.models import (
    Challenge,
    ChallengeRequest,
    ChallengeType,
    Difficulty,
    MettleResult,
    MettleSession,
    VerificationResult,
)
from pydantic import ValidationError


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
            Challenge(
                type=ChallengeType.SPEED_MATH,
                prompt="Test",
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                time_limit_ms=1000,
            )

    def test_challenge_requires_prompt(self):
        """Test that prompt is required."""
        with pytest.raises(ValidationError):
            Challenge(
                id="mtl_test",
                type=ChallengeType.SPEED_MATH,
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                time_limit_ms=1000,
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


class TestChallengeRequest:
    """Test ChallengeRequest model."""

    def test_default_difficulty(self):
        """Test default difficulty is BASIC."""
        request = ChallengeRequest()
        assert request.difficulty == Difficulty.BASIC

    def test_optional_entity_id(self):
        """Test entity_id is optional."""
        request = ChallengeRequest()
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


class TestMettleResult:
    """Test MettleResult model."""

    def test_create_mettle_result(self):
        """Test creating a METTLE result."""
        result = MettleResult(
            entity_id="agent-001",
            verified=True,
            passed=3,
            total=3,
            pass_rate=1.0,
            results=[],
        )
        assert result.verified
        assert result.pass_rate == 1.0

    def test_mettle_result_auto_issued_at(self):
        """Test that issued_at is auto-populated."""
        result = MettleResult(
            entity_id=None,
            verified=False,
            passed=0,
            total=3,
            pass_rate=0.0,
            results=[],
        )
        assert result.issued_at is not None

    def test_mettle_result_badge_optional(self):
        """Test that badge is optional."""
        result = MettleResult(
            entity_id=None,
            verified=True,
            passed=3,
            total=3,
            pass_rate=1.0,
            results=[],
        )
        assert result.badge is None


class TestMettleSession:
    """Test MettleSession model."""

    def test_create_session(self):
        """Test creating a session."""
        session = MettleSession(
            session_id="ses_abc123",
            entity_id="agent-001",
            difficulty=Difficulty.BASIC,
            challenges=[],
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
        )
        assert session.results == []
        assert session.started_at is not None
        assert not session.completed
