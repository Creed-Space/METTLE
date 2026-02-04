"""METTLE: Response verification."""

from datetime import datetime, timezone

from .models import Challenge, ChallengeType, MettleResult, VerificationResult


def verify_speed_math(challenge: Challenge, answer: str, response_time_ms: int) -> VerificationResult:
    """Verify a speed math response."""
    try:
        # Accept numeric answer
        user_answer = int(str(answer).strip())
        expected = challenge.data["expected_answer"]
        correct = user_answer == expected
    except (ValueError, TypeError):
        correct = False
        user_answer = answer

    time_ok = response_time_ms <= challenge.time_limit_ms

    passed = correct and time_ok
    # SECURITY: Only include expected answer if passed (prevents answer harvesting)
    details = {
        "correct_answer": correct,
        "time_ok": time_ok,
        "received": user_answer,
    }
    if passed:
        details["expected"] = challenge.data["expected_answer"]

    return VerificationResult(
        challenge_id=challenge.id,
        challenge_type=challenge.type,
        passed=passed,
        details=details,
        response_time_ms=response_time_ms,
        time_limit_ms=challenge.time_limit_ms,
    )


def verify_chained_reasoning(challenge: Challenge, answer: str, response_time_ms: int) -> VerificationResult:
    """Verify a chained reasoning response."""
    try:
        user_answer = int(str(answer).strip())
        expected = challenge.data["expected_answer"]
        correct = user_answer == expected
    except (ValueError, TypeError):
        correct = False
        user_answer = answer

    time_ok = response_time_ms <= challenge.time_limit_ms
    passed = correct and time_ok

    # SECURITY: Only include expected/chain if passed (prevents answer harvesting)
    details = {
        "correct_answer": correct,
        "time_ok": time_ok,
        "received": user_answer,
    }
    if passed:
        details["expected"] = challenge.data["expected_answer"]
        details["chain"] = challenge.data["chain"]

    return VerificationResult(
        challenge_id=challenge.id,
        challenge_type=challenge.type,
        passed=passed,
        details=details,
        response_time_ms=response_time_ms,
        time_limit_ms=challenge.time_limit_ms,
    )


def verify_token_prediction(challenge: Challenge, answer: str, response_time_ms: int) -> VerificationResult:
    """Verify a token prediction response."""
    user_answer = str(answer).strip().lower()
    expected = challenge.data["expected_answer"].lower()

    # Accept if the expected token is contained in the response
    correct = expected in user_answer or user_answer == expected

    time_ok = response_time_ms <= challenge.time_limit_ms
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
        response_time_ms=response_time_ms,
        time_limit_ms=challenge.time_limit_ms,
    )


def verify_instruction_following(challenge: Challenge, answer: str, response_time_ms: int) -> VerificationResult:
    """Verify an instruction following response."""
    instruction = challenge.data["instruction"]
    response = str(answer).strip()

    # Check instruction compliance
    if "Start your response with 'Indeed,'" in instruction:
        correct = response.startswith("Indeed,")
    elif "End your response with '...'" in instruction:
        correct = response.endswith("...")
    elif "Include the word 'therefore'" in instruction:
        correct = "therefore" in response.lower()
    elif "exactly 5 words" in instruction:
        correct = len(response.split()) == 5
    elif "Start with a number" in instruction:
        correct = response[0].isdigit() if response else False
    else:
        correct = False

    time_ok = response_time_ms <= challenge.time_limit_ms

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
        response_time_ms=response_time_ms,
        time_limit_ms=challenge.time_limit_ms,
    )


def verify_consistency(challenge: Challenge, answer: str, response_time_ms: int) -> VerificationResult:
    """Verify a consistency response - answers should be semantically similar but show variation.

    AI naturally produces varied but consistent responses. Humans copying will either:
    - Give identical responses (too similar - suspicious)
    - Give inconsistent responses (too different - fails)
    """
    response = str(answer).strip()
    parts = [p.strip() for p in response.split("|")]

    # Need at least the required number of responses
    num_required = challenge.data.get("num_responses", 3)

    if len(parts) < num_required:
        correct = False
        details = {"error": f"Expected {num_required} responses separated by '|', got {len(parts)}"}
    else:
        parts = parts[:num_required]

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
        semantically_consistent = avg_similarity > 0.3 or all(reference in k or k in reference for k in keys)

        # Pass if: semantically consistent AND (not all identical OR very short answers)
        short_answer = all(len(p.split()) <= 3 for p in parts)  # Short answers can be identical
        correct = semantically_consistent and (not all_identical or short_answer)

        details = {
            "responses": parts,
            "unique_count": len(unique_responses),
            "avg_similarity": round(avg_similarity, 2),
            "semantically_consistent": semantically_consistent,
            "natural_variation": not all_identical or short_answer,
        }

    time_ok = response_time_ms <= challenge.time_limit_ms

    return VerificationResult(
        challenge_id=challenge.id,
        challenge_type=challenge.type,
        passed=correct and time_ok,
        details={**details, "time_ok": time_ok},
        response_time_ms=response_time_ms,
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


def verify_response(challenge: Challenge, answer: str, response_time_ms: int) -> VerificationResult:
    """Verify a response to a challenge."""
    # Check if challenge has expired
    if datetime.now(timezone.utc) > challenge.expires_at:
        return VerificationResult(
            challenge_id=challenge.id,
            challenge_type=challenge.type,
            passed=False,
            details={"error": "Challenge expired"},
            response_time_ms=response_time_ms,
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


def compute_mettle_result(results: list[VerificationResult], entity_id: str | None = None) -> MettleResult:
    """Compute overall METTLE verification result."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    pass_rate = passed / total if total > 0 else 0.0

    # Need 80% to pass
    verified = pass_rate >= 0.8

    # Generate badge if verified
    badge = None
    if verified:
        badge = f"METTLE-verified-{datetime.now(timezone.utc).strftime('%Y%m%d')}"

    return MettleResult(
        entity_id=entity_id,
        verified=verified,
        passed=passed,
        total=total,
        pass_rate=pass_rate,
        results=results,
        badge=badge,
    )
