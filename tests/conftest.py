"""Pytest fixtures for METTLE tests."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
from fastapi.testclient import TestClient

from main import app, sessions, challenges, limiter
from mettle.models import (
    Challenge,
    ChallengeType,
    Difficulty,
    MettleSession,
    VerificationResult,
)


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_state():
    """Clear session state and rate limiter before each test."""
    sessions.clear()
    challenges.clear()
    # Reset rate limiter storage
    limiter.reset()
    yield
    sessions.clear()
    challenges.clear()
    limiter.reset()


@pytest.fixture
def sample_speed_math_challenge():
    """Create a sample speed math challenge."""
    return Challenge(
        id="mtl_test_speed",
        type=ChallengeType.SPEED_MATH,
        prompt="Calculate: 25 + 17",
        data={"expected_answer": 42, "a": 25, "b": 17, "op": "+"},
        expires_at=datetime.utcnow() + timedelta(minutes=5),
        time_limit_ms=5000,
    )


@pytest.fixture
def sample_token_challenge():
    """Create a sample token prediction challenge."""
    return Challenge(
        id="mtl_test_token",
        type=ChallengeType.TOKEN_PREDICTION,
        prompt="Complete: The quick brown ___ jumps over the lazy dog",
        data={"expected_answer": "fox"},
        expires_at=datetime.utcnow() + timedelta(minutes=5),
        time_limit_ms=5000,
    )


@pytest.fixture
def sample_instruction_challenge():
    """Create a sample instruction following challenge."""
    return Challenge(
        id="mtl_test_instruction",
        type=ChallengeType.INSTRUCTION_FOLLOWING,
        prompt="Follow this instruction: Start your response with 'Indeed,'\nThen answer: What is the capital of France?",
        data={
            "instruction": "Start your response with 'Indeed,'",
            "validator_id": "abc123",
        },
        expires_at=datetime.utcnow() + timedelta(minutes=5),
        time_limit_ms=10000,
    )


@pytest.fixture
def sample_chained_challenge():
    """Create a sample chained reasoning challenge."""
    return Challenge(
        id="mtl_test_chained",
        type=ChallengeType.CHAINED_REASONING,
        prompt="Follow these steps and give the final number:\n1. Start with 10\n2. Double it\n3. Add 10",
        data={"expected_answer": 30, "chain": [10, 20, 30], "instructions": ["Start with 10", "Double it", "Add 10"]},
        expires_at=datetime.utcnow() + timedelta(minutes=5),
        time_limit_ms=10000,
    )


@pytest.fixture
def sample_consistency_challenge():
    """Create a sample consistency challenge."""
    return Challenge(
        id="mtl_test_consistency",
        type=ChallengeType.CONSISTENCY,
        prompt="Answer this question THREE times, separated by '|':\nWhat is 2 + 2?",
        data={"question": "What is 2 + 2?", "num_responses": 3},
        expires_at=datetime.utcnow() + timedelta(minutes=5),
        time_limit_ms=15000,
    )


@pytest.fixture
def expired_challenge():
    """Create an expired challenge."""
    return Challenge(
        id="mtl_test_expired",
        type=ChallengeType.SPEED_MATH,
        prompt="Calculate: 1 + 1",
        data={"expected_answer": 2, "a": 1, "b": 1, "op": "+"},
        expires_at=datetime.utcnow() - timedelta(minutes=1),  # Already expired
        time_limit_ms=5000,
    )


@pytest.fixture
def sample_verification_results():
    """Create sample verification results."""
    return [
        VerificationResult(
            challenge_id="mtl_1",
            challenge_type=ChallengeType.SPEED_MATH,
            passed=True,
            details={"correct_answer": True, "time_ok": True},
            response_time_ms=1000,
            time_limit_ms=5000,
        ),
        VerificationResult(
            challenge_id="mtl_2",
            challenge_type=ChallengeType.TOKEN_PREDICTION,
            passed=True,
            details={"correct_answer": True, "time_ok": True},
            response_time_ms=800,
            time_limit_ms=5000,
        ),
        VerificationResult(
            challenge_id="mtl_3",
            challenge_type=ChallengeType.INSTRUCTION_FOLLOWING,
            passed=True,
            details={"instruction_followed": True, "time_ok": True},
            response_time_ms=1200,
            time_limit_ms=10000,
        ),
    ]
