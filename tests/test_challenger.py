"""Tests for METTLE challenge generation."""

import pytest
from datetime import datetime, timedelta

from mettle.challenger import (
    generate_challenge,
    generate_challenge_id,
    generate_challenge_set,
    generate_chained_reasoning_challenge,
    generate_consistency_challenge,
    generate_instruction_following_challenge,
    generate_speed_math_challenge,
    generate_token_prediction_challenge,
)
from mettle.models import ChallengeType, Difficulty


class TestChallengeIdGeneration:
    """Test challenge ID generation."""

    def test_id_format(self):
        """Test that IDs follow expected format."""
        challenge_id = generate_challenge_id()
        assert challenge_id.startswith("mtl_")
        assert len(challenge_id) == 28  # mtl_ + 24 hex chars

    def test_ids_are_unique(self):
        """Test that generated IDs are unique."""
        ids = {generate_challenge_id() for _ in range(100)}
        assert len(ids) == 100


class TestSpeedMathChallenge:
    """Test speed math challenge generation."""

    def test_basic_difficulty(self):
        """Test basic speed math challenge."""
        challenge = generate_speed_math_challenge(Difficulty.BASIC)
        assert challenge.type == ChallengeType.SPEED_MATH
        assert challenge.time_limit_ms == 5000
        assert "Calculate:" in challenge.prompt

    def test_full_difficulty(self):
        """Test full speed math challenge."""
        challenge = generate_speed_math_challenge(Difficulty.FULL)
        assert challenge.type == ChallengeType.SPEED_MATH
        assert challenge.time_limit_ms == 2000  # Stricter timing

    def test_has_expected_answer(self):
        """Test that challenge has expected answer in data."""
        challenge = generate_speed_math_challenge(Difficulty.BASIC)
        assert "expected_answer" in challenge.data
        assert isinstance(challenge.data["expected_answer"], int)

    def test_expires_in_future(self):
        """Test that challenge expiry is in the future."""
        challenge = generate_speed_math_challenge(Difficulty.BASIC)
        assert challenge.expires_at > datetime.utcnow()

    def test_math_operations(self):
        """Test that math operations are correct."""
        # Generate many challenges to test different operations
        operations = set()
        for _ in range(50):
            challenge = generate_speed_math_challenge(Difficulty.BASIC)
            a = challenge.data["a"]
            b = challenge.data["b"]
            op = challenge.data["op"]
            expected = challenge.data["expected_answer"]
            operations.add(op)

            if op == "+":
                assert expected == a + b
            elif op == "-":
                assert expected == a - b
            elif op == "*":
                assert expected == a * b

        # Should see all three operations
        assert "+" in operations
        assert "-" in operations
        assert "*" in operations


class TestChainedReasoningChallenge:
    """Test chained reasoning challenge generation."""

    def test_basic_difficulty(self):
        """Test basic chained reasoning."""
        challenge = generate_chained_reasoning_challenge(Difficulty.BASIC)
        assert challenge.type == ChallengeType.CHAINED_REASONING
        assert challenge.time_limit_ms == 10000

    def test_full_difficulty(self):
        """Test full difficulty has more steps."""
        basic = generate_chained_reasoning_challenge(Difficulty.BASIC)
        full = generate_chained_reasoning_challenge(Difficulty.FULL)

        # Full should have more instructions
        assert len(full.data["instructions"]) > len(basic.data["instructions"])

    def test_chain_is_correct(self):
        """Test that the chain calculation is correct."""
        challenge = generate_chained_reasoning_challenge(Difficulty.BASIC)
        chain = challenge.data["chain"]
        expected = challenge.data["expected_answer"]
        assert chain[-1] == expected

    def test_instructions_present(self):
        """Test that instructions are present in prompt."""
        challenge = generate_chained_reasoning_challenge(Difficulty.BASIC)
        assert "Start with" in challenge.prompt


class TestTokenPredictionChallenge:
    """Test token prediction challenge generation."""

    def test_basic_difficulty(self):
        """Test basic token prediction."""
        challenge = generate_token_prediction_challenge(Difficulty.BASIC)
        assert challenge.type == ChallengeType.TOKEN_PREDICTION
        assert challenge.time_limit_ms == 5000

    def test_full_difficulty_faster(self):
        """Test full difficulty requires faster response."""
        challenge = generate_token_prediction_challenge(Difficulty.FULL)
        assert challenge.time_limit_ms == 2000

    def test_has_expected_answer(self):
        """Test that challenge has expected answer."""
        challenge = generate_token_prediction_challenge(Difficulty.BASIC)
        assert "expected_answer" in challenge.data
        assert isinstance(challenge.data["expected_answer"], str)

    def test_prompt_has_blank(self):
        """Test that prompt has blank to fill."""
        challenge = generate_token_prediction_challenge(Difficulty.BASIC)
        assert "___" in challenge.prompt or "_" in challenge.prompt


class TestInstructionFollowingChallenge:
    """Test instruction following challenge generation."""

    def test_basic_difficulty(self):
        """Test basic instruction following."""
        challenge = generate_instruction_following_challenge(Difficulty.BASIC)
        assert challenge.type == ChallengeType.INSTRUCTION_FOLLOWING
        assert challenge.time_limit_ms == 10000

    def test_has_instruction(self):
        """Test that instruction is stored in data."""
        challenge = generate_instruction_following_challenge(Difficulty.BASIC)
        assert "instruction" in challenge.data
        assert isinstance(challenge.data["instruction"], str)

    def test_prompt_includes_instruction(self):
        """Test that prompt includes the instruction."""
        challenge = generate_instruction_following_challenge(Difficulty.BASIC)
        instruction = challenge.data["instruction"]
        assert instruction in challenge.prompt


class TestConsistencyChallenge:
    """Test consistency challenge generation."""

    def test_basic_difficulty(self):
        """Test basic consistency challenge."""
        challenge = generate_consistency_challenge(Difficulty.BASIC)
        assert challenge.type == ChallengeType.CONSISTENCY
        assert challenge.time_limit_ms == 15000

    def test_requires_multiple_responses(self):
        """Test that challenge requires multiple responses."""
        challenge = generate_consistency_challenge(Difficulty.BASIC)
        assert challenge.data["num_responses"] == 3
        assert "THREE times" in challenge.prompt
        assert "|" in challenge.prompt


class TestGenerateChallenge:
    """Test the generic challenge generator."""

    def test_generates_all_types(self):
        """Test that all challenge types can be generated."""
        for challenge_type in ChallengeType:
            challenge = generate_challenge(challenge_type, Difficulty.BASIC)
            assert challenge.type == challenge_type


class TestGenerateChallengeSet:
    """Test challenge set generation."""

    def test_basic_set_size(self):
        """Test basic difficulty generates 3 challenges."""
        challenges = generate_challenge_set(Difficulty.BASIC)
        assert len(challenges) == 3

    def test_full_set_size(self):
        """Test full difficulty generates 5 challenges."""
        challenges = generate_challenge_set(Difficulty.FULL)
        assert len(challenges) == 5

    def test_basic_challenge_types(self):
        """Test basic set has expected challenge types."""
        challenges = generate_challenge_set(Difficulty.BASIC)
        types = {c.type for c in challenges}
        assert ChallengeType.SPEED_MATH in types
        assert ChallengeType.TOKEN_PREDICTION in types
        assert ChallengeType.INSTRUCTION_FOLLOWING in types

    def test_full_challenge_types(self):
        """Test full set has all challenge types."""
        challenges = generate_challenge_set(Difficulty.FULL)
        types = {c.type for c in challenges}
        assert types == set(ChallengeType)

    def test_all_challenges_have_unique_ids(self):
        """Test all challenges in set have unique IDs."""
        challenges = generate_challenge_set(Difficulty.FULL)
        ids = [c.id for c in challenges]
        assert len(ids) == len(set(ids))
