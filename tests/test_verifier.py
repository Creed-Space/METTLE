"""Tests for METTLE response verification."""

import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from mettle.models import Challenge, ChallengeType
from mettle.verifier import (
    MAX_ANSWER_CHARS,
    compute_mettle_result,
    verify_chained_reasoning,
    verify_consistency,
    verify_instruction_following,
    verify_response,
    verify_speed_math,
    verify_token_prediction,
)


class _StringifiesAsValidAnswer:
    """Non-JSON object used to exercise the verifier's direct Python API."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:
        return self.text


def _bind_instruction(challenge: Challenge, instruction: str) -> Challenge:
    validator_id = hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:8]
    return challenge.model_copy(
        update={"data": {"instruction": instruction, "validator_id": validator_id}}
    )


def _bind_consistency(challenge: Challenge, expected: str = "4") -> Challenge:
    return challenge.model_copy(
        update={
            "data": {
                **challenge.data,
                "expected_answer": expected,
            }
        }
    )


class TestVerifySpeedMath:
    """Test speed math verification."""

    def test_correct_answer_within_time(self, sample_speed_math_challenge):
        """Test correct answer within time limit passes."""
        result = verify_speed_math(sample_speed_math_challenge, "42", 1000)
        assert result.passed
        assert result.details["correct_answer"]
        assert result.details["time_ok"]

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
        assert (
            result.details["received"] == "forty-two"
        )  # String preserved on exception

    def test_whitespace_handling(self, sample_speed_math_challenge):
        """Test that whitespace is trimmed."""
        result = verify_speed_math(sample_speed_math_challenge, "  42  ", 1000)
        assert result.passed

    def test_integer_answer_is_supported_but_stringifiable_object_is_rejected(
        self, sample_speed_math_challenge
    ):
        assert verify_speed_math(sample_speed_math_challenge, 42, 1000).passed
        result = verify_speed_math(
            sample_speed_math_challenge, _StringifiesAsValidAnswer("42"), 1000
        )
        assert result.passed is False
        assert result.details["received"] is None

    def test_oversized_integer_is_rejected_without_result_retention(
        self, sample_speed_math_challenge
    ):
        oversized = 1 << (MAX_ANSWER_CHARS * 5)
        result = verify_speed_math(sample_speed_math_challenge, oversized, 1000)
        assert result.passed is False
        assert result.details["received"] is None


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

    def test_integer_answer_is_supported_but_stringifiable_object_is_rejected(
        self, sample_chained_challenge
    ):
        assert verify_chained_reasoning(sample_chained_challenge, 30, 1000).passed
        result = verify_chained_reasoning(
            sample_chained_challenge, _StringifiesAsValidAnswer("30"), 1000
        )
        assert result.passed is False
        assert result.details["received"] is None


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

    def test_answer_wrapper_is_rejected(self, sample_token_challenge):
        """Only the requested missing token is accepted."""
        result = verify_token_prediction(
            sample_token_challenge, "The answer is fox", 1000
        )
        assert not result.passed

    def test_wrong_token(self, sample_token_challenge):
        """Test wrong token fails."""
        result = verify_token_prediction(sample_token_challenge, "cat", 1000)
        assert not result.passed

    def test_numeric_token_is_not_implicitly_stringified(self, sample_token_challenge):
        challenge = sample_token_challenge.model_copy(
            update={"data": {"expected_answer": "2"}}
        )
        result = verify_token_prediction(challenge, 2, 1000)
        assert result.passed is False
        assert result.details["received"] == ""


class TestVerifyInstructionFollowing:
    """Test instruction following verification."""

    def test_starts_with_indeed(self, sample_instruction_challenge):
        """Test 'starts with Indeed' instruction."""
        challenge = _bind_instruction(
            sample_instruction_challenge, "Start your response with 'Indeed,'"
        )
        result = verify_instruction_following(
            challenge,
            "Indeed, the capital of France is Paris.",
            1000,
        )
        assert result.passed
        assert result.details["instruction_followed"]

    def test_fails_without_indeed(self, sample_instruction_challenge):
        """Test failure when Indeed is missing."""
        challenge = _bind_instruction(
            sample_instruction_challenge, "Start your response with 'Indeed,'"
        )
        result = verify_instruction_following(
            challenge, "The capital of France is Paris.", 1000
        )
        assert not result.passed
        assert not result.details["instruction_followed"]

    def test_response_preview_truncated(self, sample_instruction_challenge):
        """Test that long responses are truncated in details."""
        challenge = _bind_instruction(
            sample_instruction_challenge, "Start your response with 'Indeed,'"
        )
        long_response = "Indeed, Paris " + "x" * 200
        result = verify_instruction_following(challenge, long_response, 1000)
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
        result = verify_instruction_following(
            unknown_challenge, "I did a backflip!", 1000
        )
        assert not result.passed
        assert not result.details["instruction_followed"]

    @pytest.mark.parametrize(
        ("instruction", "answer"),
        [
            ("End your response with '...'", "The capital is Paris..."),
            (
                "Include the word 'therefore' in your response",
                "Therefore, the capital is Paris.",
            ),
            ("Respond in exactly 5 words", "Paris is France's capital city"),
            ("Start with a number", "1: Paris is France's capital"),
        ],
    )
    def test_each_generated_instruction_requires_compliance_and_truth(
        self, sample_instruction_challenge, instruction, answer
    ):
        challenge = _bind_instruction(sample_instruction_challenge, instruction)
        result = verify_instruction_following(challenge, answer, 1000)
        assert result.passed
        assert result.details["validator_bound"] is True
        assert result.details["factual_correct"] is True

    def test_start_with_number_empty_response(self):
        """Test 'Start with a number' with empty response fails."""
        challenge = _bind_instruction(
            Challenge(
                id="mtl_test_number_empty",
                type=ChallengeType.INSTRUCTION_FOLLOWING,
                prompt="Start with a number",
                data={},
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                time_limit_ms=10000,
            ),
            "Start with a number",
        )
        result = verify_instruction_following(challenge, "", 1000)
        assert not result.passed

    def test_instruction_validator_id_cannot_be_swapped(
        self, sample_instruction_challenge
    ):
        challenge = sample_instruction_challenge.model_copy(
            update={
                "data": {
                    "instruction": "Start your response with 'Indeed,'",
                    "validator_id": "00000000",
                }
            }
        )
        result = verify_instruction_following(
            challenge, "Indeed, the capital is Paris.", 1000
        )
        assert not result.passed
        assert result.details["validator_bound"] is False

    def test_instruction_requires_factual_answer(self, sample_instruction_challenge):
        challenge = _bind_instruction(
            sample_instruction_challenge, "Start your response with 'Indeed,'"
        )
        result = verify_instruction_following(challenge, "Indeed, banana.", 1000)
        assert not result.passed
        assert result.details["rule_followed"] is True
        assert result.details["factual_correct"] is False

    def test_therefore_requires_a_complete_word(self, sample_instruction_challenge):
        challenge = _bind_instruction(
            sample_instruction_challenge,
            "Include the word 'therefore' in your response",
        )
        result = verify_instruction_following(
            challenge, "Paris is thereforeish the answer.", 1000
        )
        assert not result.passed
        assert result.details["rule_followed"] is False

    @pytest.mark.parametrize(
        "answer",
        [
            ["therefore", "Paris"],
            {"therefore": "Paris"},
            True,
        ],
    )
    def test_json_non_string_answers_are_not_interpreted_as_prose(
        self, sample_instruction_challenge, answer
    ):
        challenge = _bind_instruction(
            sample_instruction_challenge,
            "Include the word 'therefore' in your response",
        )
        result = verify_instruction_following(challenge, answer, 1000)
        assert result.passed is False
        assert result.details["response_preview"] == ""


class TestVerifyConsistency:
    """Test consistency verification."""

    def test_consistent_short_answers(self, sample_consistency_challenge):
        """The exact requested count of correct answers passes."""
        challenge = _bind_consistency(sample_consistency_challenge)
        result = verify_consistency(challenge, "4|4|4", 1000)
        assert result.passed
        assert result.details["all_answers_correct"] is True

    def test_insufficient_answers(self, sample_consistency_challenge):
        """Test insufficient number of answers fails."""
        challenge = _bind_consistency(sample_consistency_challenge)
        result = verify_consistency(challenge, "4|4", 1000)
        assert not result.passed
        assert "error" in result.details

    @pytest.mark.parametrize("answer", ["4|4|4|4", "4||4", "4|banana|4", "4,4,4"])
    def test_extra_blank_or_incorrect_answers_fail(
        self, sample_consistency_challenge, answer
    ):
        challenge = _bind_consistency(sample_consistency_challenge)
        result = verify_consistency(challenge, answer, 1000)
        assert not result.passed

    def test_case_and_terminal_punctuation_are_normalized(self):
        now = datetime.now(timezone.utc)
        challenge = Challenge(
            id="mtl_consistency_paris",
            type=ChallengeType.CONSISTENCY,
            prompt="Answer three times",
            data={"num_responses": 3, "expected_answer": "Paris"},
            issued_at=now,
            expires_at=now + timedelta(minutes=5),
            time_limit_ms=1000,
        )
        result = verify_consistency(challenge, "PARIS|Paris.|paris!", 1000)
        assert result.passed

    def test_stringifiable_object_is_not_interpreted_as_three_answers(
        self, sample_consistency_challenge
    ):
        challenge = _bind_consistency(sample_consistency_challenge)
        result = verify_consistency(challenge, _StringifiesAsValidAnswer("4|4|4"), 1000)
        assert result.passed is False
        assert result.details["error"].startswith("Answer must be a string")


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

        instruction_challenge = _bind_instruction(
            sample_instruction_challenge, "Start your response with 'Indeed,'"
        )
        result3 = verify_response(instruction_challenge, "Indeed, Paris.", 1000)
        assert result3.challenge_type == ChallengeType.INSTRUCTION_FOLLOWING

        result4 = verify_response(sample_chained_challenge, "30", 1000)
        assert result4.challenge_type == ChallengeType.CHAINED_REASONING

        consistency_challenge = _bind_consistency(sample_consistency_challenge)
        result5 = verify_response(consistency_challenge, "4|4|4", 1000)
        assert result5.challenge_type == ChallengeType.CONSISTENCY

    def test_expired_challenge_fails(self):
        """Test expired challenge fails."""
        now = datetime.now(timezone.utc)
        expired_challenge = Challenge(
            id="mtl_expired",
            type=ChallengeType.SPEED_MATH,
            prompt="Calculate: 1 + 1",
            data={"expected_answer": 2},
            issued_at=now - timedelta(minutes=2),
            expires_at=now - timedelta(minutes=1),
            time_limit_ms=5000,
        )
        result = verify_response(expired_challenge, "2", 100)
        assert not result.passed
        assert result.details.get("error") == "Challenge expired"

    @pytest.mark.parametrize(
        ("fixture_name", "answer"),
        [
            ("sample_speed_math_challenge", "42"),
            ("sample_chained_challenge", "30"),
            ("sample_token_challenge", "fox"),
            ("sample_instruction_challenge", "Indeed, Paris."),
            ("sample_consistency_challenge", "4|4|4"),
        ],
    )
    def test_every_verifier_enforces_time_domain_and_exact_boundary(
        self, request, fixture_name, answer
    ):
        challenge = request.getfixturevalue(fixture_name)
        if challenge.type == ChallengeType.INSTRUCTION_FOLLOWING:
            challenge = _bind_instruction(
                challenge, "Start your response with 'Indeed,'"
            )
        elif challenge.type == ChallengeType.CONSISTENCY:
            challenge = _bind_consistency(challenge)

        assert verify_response(challenge, answer, challenge.time_limit_ms).passed
        assert not verify_response(
            challenge, answer, challenge.time_limit_ms + 1
        ).passed
        assert not verify_response(challenge, answer, -1).passed
        assert not verify_response(challenge, answer, True).passed

    @pytest.mark.parametrize("challenge_type", list(ChallengeType))
    def test_malformed_challenge_state_fails_closed(self, challenge_type):
        now = datetime.now(timezone.utc)
        challenge = Challenge(
            id=f"mtl_malformed_{challenge_type.value}",
            type=challenge_type,
            prompt="Malformed state probe",
            data={},
            issued_at=now,
            expires_at=now + timedelta(minutes=5),
            time_limit_ms=1000,
        )

        result = verify_response(challenge, "42", 0)

        assert result.passed is False

    @pytest.mark.parametrize(
        ("fixture_name", "answer"),
        [
            ("sample_speed_math_challenge", "42"),
            ("sample_chained_challenge", "30"),
            ("sample_token_challenge", "fox"),
            ("sample_instruction_challenge", "Indeed, Paris."),
            ("sample_consistency_challenge", "4|4|4"),
        ],
    )
    def test_every_verifier_enforces_the_direct_answer_size_boundary(
        self, request, fixture_name, answer
    ):
        challenge = request.getfixturevalue(fixture_name)
        if challenge.type == ChallengeType.INSTRUCTION_FOLLOWING:
            challenge = _bind_instruction(
                challenge, "Start your response with 'Indeed,'"
            )
        elif challenge.type == ChallengeType.CONSISTENCY:
            challenge = _bind_consistency(challenge)

        exact = answer + " " * (MAX_ANSWER_CHARS - len(answer))
        assert verify_response(challenge, exact, 0).passed

        rejected = verify_response(challenge, exact + " ", 0)
        assert rejected.passed is False
        assert len(repr(rejected.details)) < 512


class TestComputeMettleResult:
    """Test METTLE result computation."""

    def test_all_passed(self, sample_verification_results):
        """Test result when all challenges passed."""
        result = compute_mettle_result(sample_verification_results, "agent-001")
        assert result.screening_passed
        assert result.verified
        assert result.credential_eligible
        assert result.passed == 3
        assert result.total == 3
        assert result.pass_rate == 1.0
        assert result.badge is None

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
        assert result.screening_passed
        assert result.verified
        assert result.credential_eligible
        assert result.pass_rate == 0.8

    def test_empty_results(self):
        """Test handling of empty results."""
        result = compute_mettle_result([], "agent-001")
        assert not result.verified
        assert result.pass_rate == 0.0

    def test_single_result_cannot_verify_a_truncated_battery(self):
        from mettle.models import VerificationResult

        only = VerificationResult(
            challenge_id="mtl_only",
            challenge_type=ChallengeType.SPEED_MATH,
            passed=True,
            response_time_ms=1,
            time_limit_ms=5000,
        )
        result = compute_mettle_result([only])
        assert result.pass_rate == 1.0
        assert result.verified is False
        assert result.credential_eligible is False

    def test_duplicate_result_ids_cannot_inflate_pass_rate(self):
        from pydantic import ValidationError
        from mettle.models import VerificationResult

        duplicate = VerificationResult(
            challenge_id="mtl_duplicate",
            challenge_type=ChallengeType.SPEED_MATH,
            passed=True,
            response_time_ms=1,
            time_limit_ms=5000,
        )
        with pytest.raises(ValidationError, match="duplicate challenge IDs"):
            compute_mettle_result([duplicate, duplicate, duplicate])

    def test_result_is_eligible_but_issuance_is_a_separate_boundary(
        self, sample_verification_results
    ):
        result = compute_mettle_result(sample_verification_results)
        assert result.screening_passed
        assert result.verified is True
        assert result.credential_eligible is True
        assert result.badge is None
        assert result.badge_info is None
