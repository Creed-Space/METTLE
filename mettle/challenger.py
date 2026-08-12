"""METTLE: Challenge generation."""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Sequence, TypeVar

from .models import Challenge, ChallengeType, Difficulty

T = TypeVar("T")


def _secure_randint(a: int, b: int) -> int:
    """Generate a cryptographically secure random integer in [a, b]."""
    return a + secrets.randbelow(b - a + 1)


def _secure_choice(seq: Sequence[T]) -> T:
    """Select a random element from sequence using cryptographic randomness."""
    return seq[secrets.randbelow(len(seq))]


def generate_challenge_id() -> str:
    """Generate a unique challenge ID."""
    return f"mtl_{secrets.token_hex(12)}"


def generate_speed_math_challenge(difficulty: Difficulty) -> Challenge:
    """Generate a speed math challenge."""
    if difficulty == Difficulty.BASIC:
        a = _secure_randint(1_000, 9_999_999)
        b = _secure_randint(1_000, 9_999_999)
        op = _secure_choice(["+", "-", "*"])
        time_limit = 2500  # 2.5s procedural response window
    else:
        a = _secure_randint(10_000_000, 90_000_000)
        b = _secure_randint(10_000_000, 90_000_000)
        op = _secure_choice(["+", "-", "*"])
        time_limit = 500  # 500ms machine-oriented response window

    if op == "+":
        answer = a + b
        prompt = f"Calculate: {a} + {b}"
    elif op == "-":
        answer = a - b
        prompt = f"Calculate: {a} - {b}"
    else:
        answer = a * b
        prompt = f"Calculate: {a} × {b}"

    return Challenge(
        id=generate_challenge_id(),
        type=ChallengeType.SPEED_MATH,
        prompt=prompt,
        data={"expected_answer": answer, "a": a, "b": b, "op": op},
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        time_limit_ms=time_limit,
    )


def generate_chained_reasoning_challenge(difficulty: Difficulty) -> Challenge:
    """Generate a chained reasoning challenge."""
    steps = 3 if difficulty == Difficulty.BASIC else 5
    time_limit = (
        3000 if difficulty == Difficulty.BASIC else 800
    )  # Basic: 3s, Full: 800ms

    seed = _secure_randint(1, 50)
    chain = [seed]
    instructions = [f"Start with {seed}"]

    for i in range(steps):
        op = _secure_choice(["double", "add_10", "subtract_5"])
        current = chain[-1]

        if op == "double":
            result = current * 2
            instructions.append("Double it")
        elif op == "add_10":
            result = current + 10
            instructions.append("Add 10")
        else:
            result = current - 5
            instructions.append("Subtract 5")

        chain.append(result)

    prompt = "Follow these steps and give the final number:\n" + "\n".join(
        f"{i + 1}. {inst}" for i, inst in enumerate(instructions)
    )

    return Challenge(
        id=generate_challenge_id(),
        type=ChallengeType.CHAINED_REASONING,
        prompt=prompt,
        data={
            "expected_answer": chain[-1],
            "chain": chain,
            "instructions": instructions,
        },
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        time_limit_ms=time_limit,
    )


def generate_token_prediction_challenge(difficulty: Difficulty) -> Challenge:
    """Generate a token prediction challenge."""
    # A fresh arithmetic token sequence avoids the finite quotation corpus used
    # by earlier releases. The rule is visible, while the concrete answer is new.
    base = _secure_randint(10_000, 99_999_999)
    step = _secure_randint(17, 99_983)
    prefix = f"K{secrets.token_hex(3)}-"
    sequence = [base + (step * index) for index in range(4)]
    expected = f"{prefix}{base + (step * 4)}"
    prompt_text = ", ".join(f"{prefix}{value}" for value in sequence)
    time_limit = (
        2000 if difficulty == Difficulty.BASIC else 400
    )  # Basic: 2s, Full: sub-second

    return Challenge(
        id=generate_challenge_id(),
        type=ChallengeType.TOKEN_PREDICTION,
        prompt=f"Complete the arithmetic token sequence: {prompt_text}, {prefix}___",
        data={"expected_answer": expected.lower(), "sequence_length": 4},
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        time_limit_ms=time_limit,
    )


def generate_instruction_following_challenge(difficulty: Difficulty) -> Challenge:
    """Generate an instruction following challenge."""
    marker = f"m{secrets.token_hex(5)}"
    word_count = _secure_randint(5, 10)
    starting_digit = str(_secure_randint(1, 9))
    instruction_kind = _secure_choice(
        ["prefix", "suffix", "include", "exact_words", "start_digit"]
    )
    instructions = {
        "prefix": f"Start your response with the exact token '{marker}'",
        "suffix": f"End your response with the exact token '{marker}'",
        "include": f"Include the exact token '{marker}' as a separate word",
        "exact_words": (
            f"Respond in exactly {word_count} whitespace-separated words and "
            f"include '{marker}' as one word"
        ),
        "start_digit": (
            f"Start your response with the digit {starting_digit} and include "
            f"'{marker}' as a separate word"
        ),
    }
    instruction = instructions[instruction_kind]
    time_limit = (
        3000 if difficulty == Difficulty.BASIC else 600
    )  # Basic: 3s, Full: 600ms

    # Store a short stable validator identifier for serialization.
    validator_id = hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:8]

    return Challenge(
        id=generate_challenge_id(),
        type=ChallengeType.INSTRUCTION_FOLLOWING,
        prompt=f"Follow this instruction: {instruction}\nThen answer: What is the capital of France?",
        data={
            "instruction": instruction,
            "instruction_kind": instruction_kind,
            "marker": marker,
            "word_count": word_count,
            "starting_digit": starting_digit,
            "validator_id": validator_id,
        },
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        time_limit_ms=time_limit,
    )


def generate_consistency_challenge(difficulty: Difficulty) -> Challenge:
    """Generate a consistency challenge - answer must be consistent across attempts."""
    questions = [
        "What is 2 + 2?",
        "What color is the sky on a clear day?",
        "How many legs does a dog have?",
        "What is the capital of France?",
    ]

    question = _secure_choice(questions)
    time_limit = (
        3500 if difficulty == Difficulty.BASIC else 1000
    )  # Basic: 3.5s, Full: 1s

    return Challenge(
        id=generate_challenge_id(),
        type=ChallengeType.CONSISTENCY,
        prompt=f"Answer this question THREE times, separated by '|':\n{question}",
        data={"question": question, "num_responses": 3},
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        time_limit_ms=time_limit,
    )


def generate_challenge(
    challenge_type: ChallengeType, difficulty: Difficulty
) -> Challenge:
    """Generate a challenge of the specified type."""
    generators = {
        ChallengeType.SPEED_MATH: generate_speed_math_challenge,
        ChallengeType.CHAINED_REASONING: generate_chained_reasoning_challenge,
        ChallengeType.TOKEN_PREDICTION: generate_token_prediction_challenge,
        ChallengeType.INSTRUCTION_FOLLOWING: generate_instruction_following_challenge,
        ChallengeType.CONSISTENCY: generate_consistency_challenge,
    }
    return generators[challenge_type](difficulty)


def generate_challenge_set(difficulty: Difficulty) -> list[Challenge]:
    """Generate a full set of challenges for verification."""
    if difficulty == Difficulty.BASIC:
        # Basic: 3 challenges
        types = [
            ChallengeType.SPEED_MATH,
            ChallengeType.TOKEN_PREDICTION,
            ChallengeType.INSTRUCTION_FOLLOWING,
        ]
    else:
        # Full: all 5 challenge types
        types = list(ChallengeType)

    return [generate_challenge(t, difficulty) for t in types]
