"""METTLE: Response verification."""

from datetime import datetime, timezone
from .models import Challenge, ChallengeType, MettleResult, VerificationResult

MAX_ANSWER_CHARS = 1024
_MAX_NUMERIC_ANSWER_BITS = MAX_ANSWER_CHARS * 4


def _bounded_text_answer(answer: object) -> str | None:
    if type(answer) is not str or len(answer) > MAX_ANSWER_CHARS:
        return None
    return answer.strip()


def _parse_integer_answer(answer: object) -> tuple[int | str | None, bool]:
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
    return type(response_time_ms) is int and 0 <= response_time_ms <= time_limit_ms


def _recordable_response_time(response_time_ms: object) -> int:
    return (
        response_time_ms
        if type(response_time_ms) is int and response_time_ms >= 0
        else 0
    )


def verify_speed_math(
    challenge: Challenge, answer: object, response_time_ms: object
) -> VerificationResult:
    """Verify a speed math response."""
    user_answer, answer_is_integer = _parse_integer_answer(answer)
    expected = challenge.data.get("expected_answer")
    correct = answer_is_integer and type(expected) is int and user_answer == expected
    time_ok = _time_is_valid(response_time_ms, challenge.time_limit_ms)

    passed = correct and time_ok
    # Public verification details report the verdict, never the server-held answer.
    details = {
        "correct_answer": correct,
        "time_ok": time_ok,
        "received": user_answer,
    }
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
    user_answer, answer_is_integer = _parse_integer_answer(answer)
    expected = challenge.data.get("expected_answer")
    correct = answer_is_integer and type(expected) is int and user_answer == expected
    time_ok = _time_is_valid(response_time_ms, challenge.time_limit_ms)
    passed = correct and time_ok

    # Public verification details report the verdict, never the answer or chain.
    details = {
        "correct_answer": correct,
        "time_ok": time_ok,
        "received": user_answer,
    }
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
    bounded = _bounded_text_answer(answer)
    user_answer = bounded.casefold() if bounded is not None else ""
    raw_expected = challenge.data.get("expected_answer")
    expected = raw_expected.casefold() if type(raw_expected) is str else ""

    # The challenge asks for one missing token. Requiring the exact token keeps
    # a caller from submitting a list of every possible completion.
    correct = bounded is not None and bool(expected) and user_answer == expected

    time_ok = _time_is_valid(response_time_ms, challenge.time_limit_ms)
    passed = correct and time_ok

    # Public verification details report the verdict, never the expected token.
    details = {
        "correct_answer": correct,
        "time_ok": time_ok,
        "received": user_answer,
    }
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
    bounded = _bounded_text_answer(answer)
    response = bounded or ""
    instruction_kind = challenge.data.get("instruction_kind")
    marker = challenge.data.get("marker")

    # Check instruction compliance
    if instruction_kind == "prefix" and isinstance(marker, str):
        correct = response.startswith(marker)
    elif instruction_kind == "suffix" and isinstance(marker, str):
        correct = response.endswith(marker)
    elif instruction_kind == "include" and isinstance(marker, str):
        correct = marker in response.split()
    elif instruction_kind == "exact_words" and isinstance(marker, str):
        correct = (
            len(response.split()) == challenge.data.get("word_count")
            and marker in response.split()
        )
    elif instruction_kind == "start_digit" and isinstance(marker, str):
        correct = response.startswith(str(challenge.data.get("starting_digit"))) and (
            marker in response.split()
        )
    elif (
        isinstance(instruction, str)
        and "Start your response with 'Indeed,'" in instruction
    ):
        correct = response.startswith("Indeed,")
    elif isinstance(instruction, str) and "End your response with '...'" in instruction:
        correct = response.endswith("...")
    elif isinstance(instruction, str) and "Include the word 'therefore'" in instruction:
        correct = "therefore" in response.lower()
    elif isinstance(instruction, str) and "exactly 5 words" in instruction:
        correct = len(response.split()) == 5
    elif isinstance(instruction, str) and "Start with a number" in instruction:
        correct = response[0].isdigit() if response else False
    else:
        correct = False

    correct = bounded is not None and correct
    time_ok = _time_is_valid(response_time_ms, challenge.time_limit_ms)

    return VerificationResult(
        challenge_id=challenge.id,
        challenge_type=challenge.type,
        passed=correct and time_ok,
        details={
            "instruction_followed": correct,
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
    """Verify a consistency response - answers should be semantically similar but show variation.

    AI naturally produces varied but consistent responses. Humans copying will either:
    - Give identical responses (too similar - suspicious)
    - Give inconsistent responses (too different - fails)
    """
    bounded = _bounded_text_answer(answer)
    response = bounded or ""
    parts = [p.strip() for p in response.split("|")]

    # Need at least the required number of responses
    num_required = challenge.data.get("num_responses", 3)
    details: dict[str, object]

    if type(num_required) is not int or not 1 <= num_required <= 10:
        correct = False
        details = {"error": "Malformed consistency challenge data"}
    elif (
        bounded is None or len(parts) != num_required or any(not part for part in parts)
    ):
        correct = False
        details = {
            "error": f"Expected {num_required} responses separated by '|', got {len(parts)}"
        }
    else:
        # Extract key content words
        def extract_key(s: str) -> str:
            s = s.lower().strip().rstrip(".!?")
            for word in ["the", "a", "an", "is", "it", "its", "i", "think"]:
                s = s.replace(f" {word} ", " ")
            return s.strip()

        keys = [extract_key(p) for p in parts]

        # Check for suspicious exact duplicates (human copy-paste)
        unique_responses = set(p.lower().strip() for p in parts)
        all_identical = len(unique_responses) == 1

        # Check semantic consistency
        reference = keys[0]
        similarities = [_simple_similarity(reference, k) for k in keys[1:]]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0

        # AI behavior: varied phrasing (not identical) but consistent meaning (similar)
        # Human copy-paste: all identical
        # Human guessing: inconsistent
        semantically_consistent = avg_similarity > 0.3 or all(
            reference in k or k in reference for k in keys
        )

        # Pass if: semantically consistent AND (not all identical OR very short answers)
        short_answer = all(
            len(p.split()) <= 3 for p in parts
        )  # Short answers can be identical
        correct = semantically_consistent and (not all_identical or short_answer)

        details = {
            "responses": parts,
            "unique_count": len(unique_responses),
            "avg_similarity": round(avg_similarity, 2),
            "semantically_consistent": semantically_consistent,
            "natural_variation": not all_identical or short_answer,
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


def _simple_similarity(a: str, b: str) -> float:
    """Simple word overlap similarity."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


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
    verified = pass_rate >= 0.8

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
