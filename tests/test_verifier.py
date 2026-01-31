"""Tests for METTLE response verification."""

import pytest
from datetime import datetime, timedelta, timezone

from mettle.models import ChallengeType
from mettle.verifier import (
    _simple_similarity,
    compute_mettle_result,
    verify_chained_reasoning,
    verify_consistency,
    verify_instruction_following,
    verify_response,
    verify_speed_math,
    verify_token_prediction,
)


class TestVerifySpeedMath:
    """Test speed math verification."""

    def test_correct_answer_within_time(self, sample_speed_math_challenge):
        """Test correct answer within time limit passes."""
        result = verify_speed_math(sample_speed_math_challenge, "42", 1000)
        assert result.passed
        assert result.details["correct_answer"]
        assert result.details["time_ok"]

    def test_correct_answer_over_time(self, sample_speed_math_challenge):
        """Test correct answer over time limit fails."""
        result = verify_speed_math(sample_speed_math_challenge, "42", 10000)
        assert not result.passed
        assert result.details["correct_answer"]
        assert not result.details["time_ok"]

    def test_wrong_answer(self, sample_speed_math_challenge):
        """Test wrong answer fails."""
        result = verify_speed_math(sample_speed_math_challenge, "100", 1000)
        assert not result.passed
        assert not result.details["correct_answer"]

    def test_non_numeric_answer(self, sample_speed_math_challenge):
        """Test non-numeric answer fails gracefully."""
        result = verify_speed_math(sample_speed_math_challenge, "forty-two", 1000)
        assert not result.passed
        assert not result.details["correct_answer"]
        assert result.details["received"] == "forty-two"  # String preserved on exception

    def test_whitespace_handling(self, sample_speed_math_challenge):
        """Test that whitespace is trimmed."""
        result = verify_speed_math(sample_speed_math_challenge, "  42  ", 1000)
        assert result.passed


class TestVerifyChainedReasoning:
    """Test chained reasoning verification."""

    def test_correct_answer(self, sample_chained_challenge):
        """Test correct chained answer passes."""
        result = verify_chained_reasoning(sample_chained_challenge, "30", 1000)
        assert result.passed

    def test_wrong_answer(self, sample_chained_challenge):
        """Test wrong answer fails."""
        result = verify_chained_reasoning(sample_chained_challenge, "25", 1000)
        assert not result.passed

    def test_chain_in_details(self, sample_chained_challenge):
        """Test that chain is included in details."""
        result = verify_chained_reasoning(sample_chained_challenge, "30", 1000)
        assert "chain" in result.details

    def test_non_numeric_answer(self, sample_chained_challenge):
        """Test non-numeric answer fails gracefully."""
        result = verify_chained_reasoning(sample_chained_challenge, "thirty", 1000)
        assert not result.passed
        assert not result.details["correct_answer"]
        assert result.details["received"] == "thirty"  # String preserved on exception


class TestVerifyTokenPrediction:
    """Test token prediction verification."""

    def test_exact_match(self, sample_token_challenge):
        """Test exact match passes."""
        result = verify_token_prediction(sample_token_challenge, "fox", 1000)
        assert result.passed

    def test_case_insensitive(self, sample_token_challenge):
        """Test case insensitive matching."""
        result = verify_token_prediction(sample_token_challenge, "FOX", 1000)
        assert result.passed

    def test_contained_in_response(self, sample_token_challenge):
        """Test token contained in response passes."""
        result = verify_token_prediction(sample_token_challenge, "The answer is fox", 1000)
        assert result.passed

    def test_wrong_token(self, sample_token_challenge):
        """Test wrong token fails."""
        result = verify_token_prediction(sample_token_challenge, "cat", 1000)
        assert not result.passed


class TestVerifyInstructionFollowing:
    """Test instruction following verification."""

    def test_starts_with_indeed(self, sample_instruction_challenge):
        """Test 'starts with Indeed' instruction."""
        result = verify_instruction_following(
            sample_instruction_challenge,
            "Indeed, the capital of France is Paris.",
            1000
        )
        assert result.passed
        assert result.details["instruction_followed"]

    def test_fails_without_indeed(self, sample_instruction_challenge):
        """Test failure when Indeed is missing."""
        result = verify_instruction_following(
            sample_instruction_challenge,
            "The capital of France is Paris.",
            1000
        )
        assert not result.passed
        assert not result.details["instruction_followed"]

    def test_response_preview_truncated(self, sample_instruction_challenge):
        """Test that long responses are truncated in details."""
        long_response = "Indeed, " + "x" * 200
        result = verify_instruction_following(
            sample_instruction_challenge,
            long_response,
            1000
        )
        assert len(result.details["response_preview"]) <= 100

    def test_unknown_instruction_fails(self):
        """Test that unknown instruction types fail."""
        from mettle.models import Challenge, ChallengeType

        unknown_challenge = Challenge(
            id="mtl_test_unknown",
            type=ChallengeType.INSTRUCTION_FOLLOWING,
            prompt="Follow this weird instruction",
            data={
                "instruction": "Do a backflip while answering",  # Unknown instruction
                "validator_id": "xyz",
            },
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            time_limit_ms=10000,
        )
        result = verify_instruction_following(unknown_challenge, "I did a backflip!", 1000)
        assert not result.passed
        assert not result.details["instruction_followed"]

    def test_end_with_ellipsis_instruction(self):
        """Test 'End your response with ...' instruction."""
        from mettle.models import Challenge, ChallengeType

        challenge = Challenge(
            id="mtl_test_ellipsis",
            type=ChallengeType.INSTRUCTION_FOLLOWING,
            prompt="End your response with '...'",
            data={"instruction": "End your response with '...'", "validator_id": "x"},
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            time_limit_ms=10000,
        )
        result = verify_instruction_following(challenge, "The answer is unknown...", 1000)
        assert result.passed
        assert result.details["instruction_followed"]

    def test_include_therefore_instruction(self):
        """Test 'Include the word therefore' instruction."""
        from mettle.models import Challenge, ChallengeType

        challenge = Challenge(
            id="mtl_test_therefore",
            type=ChallengeType.INSTRUCTION_FOLLOWING,
            prompt="Include the word 'therefore'",
            data={"instruction": "Include the word 'therefore'", "validator_id": "x"},
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            time_limit_ms=10000,
        )
        result = verify_instruction_following(challenge, "Therefore, the sky is blue.", 1000)
        assert result.passed
        assert result.details["instruction_followed"]

    def test_exactly_five_words_instruction(self):
        """Test 'exactly 5 words' instruction."""
        from mettle.models import Challenge, ChallengeType

        challenge = Challenge(
            id="mtl_test_5words",
            type=ChallengeType.INSTRUCTION_FOLLOWING,
            prompt="Respond with exactly 5 words",
            data={"instruction": "Respond with exactly 5 words", "validator_id": "x"},
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            time_limit_ms=10000,
        )
        result = verify_instruction_following(challenge, "One two three four five", 1000)
        assert result.passed
        assert result.details["instruction_followed"]

    def test_start_with_number_instruction(self):
        """Test 'Start with a number' instruction."""
        from mettle.models import Challenge, ChallengeType

        challenge = Challenge(
            id="mtl_test_number",
            type=ChallengeType.INSTRUCTION_FOLLOWING,
            prompt="Start with a number",
            data={"instruction": "Start with a number", "validator_id": "x"},
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            time_limit_ms=10000,
        )
        result = verify_instruction_following(challenge, "42 is the answer", 1000)
        assert result.passed
        assert result.details["instruction_followed"]

    def test_start_with_number_empty_response(self):
        """Test 'Start with a number' with empty response fails."""
        from mettle.models import Challenge, ChallengeType

        challenge = Challenge(
            id="mtl_test_number_empty",
            type=ChallengeType.INSTRUCTION_FOLLOWING,
            prompt="Start with a number",
            data={"instruction": "Start with a number", "validator_id": "x"},
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            time_limit_ms=10000,
        )
        result = verify_instruction_following(challenge, "", 1000)
        assert not result.passed


class TestVerifyConsistency:
    """Test consistency verification."""

    def test_consistent_short_answers(self, sample_consistency_challenge):
        """Test short consistent answers pass (short answers can be identical)."""
        result = verify_consistency(
            sample_consistency_challenge,
            "4|4|4",
            1000
        )
        assert result.passed  # Short answers are allowed to be identical
        assert result.details["semantically_consistent"]

    def test_varied_but_consistent(self, sample_consistency_challenge):
        """Test varied but semantically consistent answers."""
        result = verify_consistency(
            sample_consistency_challenge,
            "4|four|4",
            1000
        )
        assert result.passed
        assert result.details["semantically_consistent"]
        assert result.details["natural_variation"]

    def test_insufficient_answers(self, sample_consistency_challenge):
        """Test insufficient number of answers fails."""
        result = verify_consistency(
            sample_consistency_challenge,
            "4|4",  # Only 2, needs 3
            1000
        )
        assert not result.passed
        assert "error" in result.details


class TestSimpleSimilarity:
    """Test simple similarity function."""

    def test_identical_strings(self):
        """Test identical strings have similarity 1.0."""
        assert _simple_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        """Test no overlap has similarity 0.0."""
        assert _simple_similarity("hello", "goodbye") == 0.0

    def test_partial_overlap(self):
        """Test partial overlap has intermediate similarity."""
        sim = _simple_similarity("hello world", "hello there")
        assert 0 < sim < 1

    def test_empty_string(self):
        """Test empty strings return 0.0."""
        assert _simple_similarity("", "hello") == 0.0
        assert _simple_similarity("hello", "") == 0.0


class TestVerifyResponse:
    """Test the main verify_response dispatcher."""

    def test_routes_to_correct_verifier(
        self,
        sample_speed_math_challenge,
        sample_token_challenge,
        sample_instruction_challenge,
        sample_chained_challenge,
        sample_consistency_challenge,
    ):
        """Test that responses are routed to correct verifier."""
        result1 = verify_response(sample_speed_math_challenge, "42", 1000)
        assert result1.challenge_type == ChallengeType.SPEED_MATH

        result2 = verify_response(sample_token_challenge, "fox", 1000)
        assert result2.challenge_type == ChallengeType.TOKEN_PREDICTION

        result3 = verify_response(sample_instruction_challenge, "Indeed, Paris.", 1000)
        assert result3.challenge_type == ChallengeType.INSTRUCTION_FOLLOWING

        result4 = verify_response(sample_chained_challenge, "30", 1000)
        assert result4.challenge_type == ChallengeType.CHAINED_REASONING

        result5 = verify_response(sample_consistency_challenge, "4|4|4", 1000)
        assert result5.challenge_type == ChallengeType.CONSISTENCY

    def test_expired_challenge_fails(self, expired_challenge):
        """Test expired challenge fails."""
        result = verify_response(expired_challenge, "2", 100)
        assert not result.passed
        assert result.details.get("error") == "Challenge expired"


class TestComputeMettleResult:
    """Test METTLE result computation."""

    def test_all_passed(self, sample_verification_results):
        """Test result when all challenges passed."""
        result = compute_mettle_result(sample_verification_results, "agent-001")
        assert result.verified
        assert result.passed == 3
        assert result.total == 3
        assert result.pass_rate == 1.0
        assert result.badge is not None

    def test_not_enough_passed(self):
        """Test result when not enough challenges passed."""
        from mettle.models import ChallengeType, VerificationResult

        results = [
            VerificationResult(
                challenge_id="mtl_1",
                challenge_type=ChallengeType.SPEED_MATH,
                passed=True,
                details={},
                response_time_ms=1000,
                time_limit_ms=5000,
            ),
            VerificationResult(
                challenge_id="mtl_2",
                challenge_type=ChallengeType.TOKEN_PREDICTION,
                passed=False,  # Failed
                details={},
                response_time_ms=1000,
                time_limit_ms=5000,
            ),
            VerificationResult(
                challenge_id="mtl_3",
                challenge_type=ChallengeType.INSTRUCTION_FOLLOWING,
                passed=False,  # Failed
                details={},
                response_time_ms=1000,
                time_limit_ms=10000,
            ),
        ]

        result = compute_mettle_result(results, "agent-001")
        assert not result.verified  # 1/3 = 33% < 80%
        assert result.passed == 1
        assert result.total == 3
        assert result.badge is None

    def test_threshold_at_80_percent(self):
        """Test that 80% threshold is enforced."""
        from mettle.models import ChallengeType, VerificationResult

        # 4 out of 5 = 80% should pass
        results = [
            VerificationResult(
                challenge_id=f"mtl_{i}",
                challenge_type=ChallengeType.SPEED_MATH,
                passed=(i < 4),  # First 4 pass
                details={},
                response_time_ms=1000,
                time_limit_ms=5000,
            )
            for i in range(5)
        ]

        result = compute_mettle_result(results)
        assert result.verified
        assert result.pass_rate == 0.8

    def test_empty_results(self):
        """Test handling of empty results."""
        result = compute_mettle_result([], "agent-001")
        assert not result.verified
        assert result.pass_rate == 0.0

    def test_badge_format(self, sample_verification_results):
        """Test badge format when verified."""
        result = compute_mettle_result(sample_verification_results)
        assert result.badge is not None
        assert result.badge.startswith("METTLE-verified-")
        # Should contain date
        assert datetime.now(timezone.utc).strftime("%Y%m%d") in result.badge
