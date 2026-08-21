"""METTLE: Response verification."""

import hashlib
import re
from datetime import datetime, timezone

from .models import Challenge, ChallengeType, MettleResult, VerificationResult


MAX_ANSWER_CHARS = 1024
_MAX_NUMERIC_ANSWER_BITS = MAX_ANSWER_CHARS * 4

_KNOWN_INSTRUCTIONS = {
    "Start your response with 'Indeed,'",
    "End your response with '...'",
    "Include the word 'therefore' in your response",
    "Respond in exactly 5 words",
    "Start with a number",
}


def _bounded_text_answer(answer: object) -> str | None:
    """Return a bounded plain string, rejecting implicit object stringification."""
    if type(answer) is not str or len(answer) > MAX_ANSWER_CHARS:
        return None
    return answer.strip()


def _parse_integer_answer(answer: object) -> tuple[int | str | None, bool]:
    """Parse bounded text or a real integer without accepting bools or objects."""
    if type(answer) is int:
        if answer.bit_length() > _MAX_NUMERIC_ANSWER_BITS:
            return None, False
        return answer, True

    text = _bounded_text_answer(answer)
    if text is None:
        return None, False
    try:
        return int(text), True
    except ValueError:
        return text, False


def _time_is_valid(response_time_ms: object, time_limit_ms: int) -> bool:
    """Accept integer elapsed times from zero through the exact limit."""
    return type(response_time_ms) is int and 0 <= response_time_ms <= time_limit_ms


def _recordable_response_time(response_time_ms: object) -> int:
    """Keep invalid caller values out of the validated result model."""
    if type(response_time_ms) is int and response_time_ms >= 0:
        return response_time_ms
    return 0


def verify_speed_math(
    challenge: Challenge, answer: object, response_time_ms: object
) -> VerificationResult:
    """Verify a speed math response."""
    expected = challenge.data.get("expected_answer")
    user_answer, answer_is_integer = _parse_integer_answer(answer)
    correct = answer_is_integer and type(expected) is int and user_answer == expected

    time_ok = _time_is_valid(response_time_ms, challenge.time_limit_ms)

    passed = correct and time_ok
    # SECURITY: Only include expected answer if passed (prevents answer harvesting)
    details: dict[str, object] = {
        "correct_answer": correct,
        "time_ok": time_ok,
        "received": user_answer,
    }
    if passed:
        details["expected"] = expected

    return VerificationResult(
        challenge_id=challenge.id,
        challenge_type=challenge.type,
        passed=passed,
        details=details,
        response_time_ms=_recordable_response_time(response_time_ms),
        time_limit_ms=challenge.time_limit_ms,
    )


def verify_chained_reasoning(
    challenge: Challenge, answer: object, response_time_ms: object
) -> VerificationResult:
    """Verify a chained reasoning response."""
    expected = challenge.data.get("expected_answer")
    chain = challenge.data.get("chain")
    valid_chain = (
        isinstance(chain, list)
        and bool(chain)
        and all(type(value) is int for value in chain)
        and type(expected) is int
        and chain[-1] == expected
    )
    user_answer, answer_is_integer = _parse_integer_answer(answer)
    correct = answer_is_integer and valid_chain and user_answer == expected

    time_ok = _time_is_valid(response_time_ms, challenge.time_limit_ms)
    passed = correct and time_ok

    # SECURITY: Only include expected/chain if passed (prevents answer harvesting)
    details: dict[str, object] = {
        "correct_answer": correct,
        "time_ok": time_ok,
        "received": user_answer,
    }
    if passed:
        details["expected"] = expected
        details["chain"] = chain

    return VerificationResult(
        challenge_id=challenge.id,
        challenge_type=challenge.type,
        passed=passed,
        details=details,
        response_time_ms=_recordable_response_time(response_time_ms),
        time_limit_ms=challenge.time_limit_ms,
    )


def verify_token_prediction(
    challenge: Challenge, answer: object, response_time_ms: object
) -> VerificationResult:
    """Verify a token prediction response."""
    raw_answer = _bounded_text_answer(answer)
    user_answer = raw_answer.casefold() if raw_answer is not None else ""
    raw_expected = challenge.data.get("expected_answer")
    if (
        type(raw_expected) is str
        and len(raw_expected) <= MAX_ANSWER_CHARS
        and len(raw_expected.split()) == 1
    ):
        expected = raw_expected.strip().casefold()
        valid_expected = bool(expected)
    else:
        expected = ""
        valid_expected = False

    # The challenge asks for one missing token. Requiring the exact token keeps
    # a caller from submitting a list of every possible completion.
    correct = raw_answer is not None and valid_expected and user_answer == expected

    time_ok = _time_is_valid(response_time_ms, challenge.time_limit_ms)
    passed = correct and time_ok

    # SECURITY: Only include expected if passed (prevents answer harvesting)
    details = {
        "correct_answer": correct,
        "time_ok": time_ok,
        "received": user_answer,
    }
    if passed:
        details["expected"] = expected

    return VerificationResult(
        challenge_id=challenge.id,
        challenge_type=challenge.type,
        passed=passed,
        details=details,
        response_time_ms=_recordable_response_time(response_time_ms),
        time_limit_ms=challenge.time_limit_ms,
    )


def verify_instruction_following(
    challenge: Challenge, answer: object, response_time_ms: object
) -> VerificationResult:
    """Verify an instruction following response."""
    instruction = challenge.data.get("instruction")
    validator_id = challenge.data.get("validator_id")
    bounded_answer = _bounded_text_answer(answer)
    response = bounded_answer if bounded_answer is not None else ""

    validator_bound = False
    if isinstance(instruction, str) and instruction in _KNOWN_INSTRUCTIONS:
        expected_validator = hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:8]
        validator_bound = validator_id == expected_validator

    if instruction == "Start your response with 'Indeed,'":
        rule_followed = response.startswith("Indeed,")
    elif instruction == "End your response with '...'":
        rule_followed = response.endswith("...")
    elif instruction == "Include the word 'therefore' in your response":
        rule_followed = re.search(r"\btherefore\b", response, re.IGNORECASE) is not None
    elif instruction == "Respond in exactly 5 words":
        rule_followed = len(response.split()) == 5
    elif instruction == "Start with a number":
        rule_followed = bool(response) and response[0].isdigit()
    else:
        rule_followed = False

    # Every generated prompt also asks the factual question about France.
    factual_correct = re.search(r"\bparis\b", response, re.IGNORECASE) is not None
    correct = (
        bounded_answer is not None
        and validator_bound
        and rule_followed
        and factual_correct
    )

    time_ok = _time_is_valid(response_time_ms, challenge.time_limit_ms)

    return VerificationResult(
        challenge_id=challenge.id,
        challenge_type=challenge.type,
        passed=correct and time_ok,
        details={
            "instruction_followed": correct,
            "validator_bound": validator_bound,
            "rule_followed": rule_followed,
            "factual_correct": factual_correct,
            "time_ok": time_ok,
            "instruction": instruction,
            "response_preview": response[:50],  # Truncated to limit info disclosure
        },
        response_time_ms=_recordable_response_time(response_time_ms),
        time_limit_ms=challenge.time_limit_ms,
    )


def verify_consistency(
    challenge: Challenge, answer: object, response_time_ms: object
) -> VerificationResult:
    """Require the exact requested count of independently correct answers."""
    response = _bounded_text_answer(answer)

    num_required = challenge.data.get("num_responses", 3)
    expected = challenge.data.get("expected_answer")
    details: dict[str, object]

    valid_challenge = (
        type(num_required) is int
        and 1 <= num_required <= 10
        and type(expected) is str
        and len(expected) <= MAX_ANSWER_CHARS
        and bool(expected.strip())
    )
    if response is None:
        correct = False
        details = {
            "error": f"Answer must be a string of at most {MAX_ANSWER_CHARS} characters"
        }
    elif not valid_challenge:
        correct = False
        details = {"error": "Malformed consistency challenge data"}
    else:
        parts = [part.strip() for part in response.split("|")]
        if len(parts) != num_required or any(not part for part in parts):
            correct = False
            details = {
                "error": (
                    f"Expected exactly {num_required} nonblank responses "
                    "separated by '|'"
                )
            }
        else:
            assert isinstance(expected, str)

            def normalize(value: str) -> str:
                return value.strip(" \t\r\n.!?,;:'\"").casefold()

            expected_normalized = normalize(expected)
            correctness = [normalize(part) == expected_normalized for part in parts]
            correct = all(correctness)
            details = {
                "responses": parts,
                "response_count": len(parts),
                "all_answers_correct": correct,
            }

    time_ok = _time_is_valid(response_time_ms, challenge.time_limit_ms)

    return VerificationResult(
        challenge_id=challenge.id,
        challenge_type=challenge.type,
        passed=correct and time_ok,
        details={**details, "time_ok": time_ok},
        response_time_ms=_recordable_response_time(response_time_ms),
        time_limit_ms=challenge.time_limit_ms,
    )


def verify_response(
    challenge: Challenge, answer: object, response_time_ms: object
) -> VerificationResult:
    """Verify a response to a challenge."""
    # Check if challenge has expired
    if datetime.now(timezone.utc) > challenge.expires_at:
        return VerificationResult(
            challenge_id=challenge.id,
            challenge_type=challenge.type,
            passed=False,
            details={"error": "Challenge expired"},
            response_time_ms=_recordable_response_time(response_time_ms),
            time_limit_ms=challenge.time_limit_ms,
        )

    verifiers = {
        ChallengeType.SPEED_MATH: verify_speed_math,
        ChallengeType.CHAINED_REASONING: verify_chained_reasoning,
        ChallengeType.TOKEN_PREDICTION: verify_token_prediction,
        ChallengeType.INSTRUCTION_FOLLOWING: verify_instruction_following,
        ChallengeType.CONSISTENCY: verify_consistency,
    }

    return verifiers[challenge.type](challenge, answer, response_time_ms)


def compute_mettle_result(
    results: list[VerificationResult], entity_id: str | None = None
) -> MettleResult:
    """Compute overall METTLE verification result."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    pass_rate = passed / total if total > 0 else 0.0

    # Like a conventional CAPTCHA, METTLE is a probabilistic gate. ``verified``
    # means that this challenge session met the configured threshold. It does
    # not claim that the self-asserted entity identifier is a proven identity.
    # Three is the smallest generated challenge battery. A truncated or
    # corrupted result list cannot confer verification by passing one item.
    verified = total >= 3 and pass_rate >= 0.8

    return MettleResult(
        entity_id=entity_id,
        verified=verified,
        screening_passed=verified,
        credential_eligible=verified,
        passed=passed,
        total=total,
        pass_rate=pass_rate,
        results=results,
        badge=None,
        badge_info=None,
    )
