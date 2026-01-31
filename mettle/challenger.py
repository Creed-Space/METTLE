"""METTLE: Challenge generation."""

import hashlib
import random
import secrets
from datetime import datetime, timedelta, timezone

from .models import Challenge, ChallengeType, Difficulty


def generate_challenge_id() -> str:
    """Generate a unique challenge ID."""
    return f"mtl_{secrets.token_hex(12)}"


def generate_speed_math_challenge(difficulty: Difficulty) -> Challenge:
    """Generate a speed math challenge."""
    if difficulty == Difficulty.BASIC:
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        op = random.choice(["+", "-", "*"])
        time_limit = 800  # 800ms - fast but achievable for AI
    else:
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        op = random.choice(["+", "-", "*"])
        time_limit = 500  # 500ms - requires native computation

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
    time_limit = 1500 if difficulty == Difficulty.BASIC else 800  # Tighter limits

    seed = random.randint(1, 50)
    chain = [seed]
    instructions = [f"Start with {seed}"]

    for i in range(steps):
        op = random.choice(["double", "add_10", "subtract_5"])
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
        f"{i+1}. {inst}" for i, inst in enumerate(instructions)
    )

    return Challenge(
        id=generate_challenge_id(),
        type=ChallengeType.CHAINED_REASONING,
        prompt=prompt,
        data={"expected_answer": chain[-1], "chain": chain, "instructions": instructions},
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        time_limit_ms=time_limit,
    )


def generate_token_prediction_challenge(difficulty: Difficulty) -> Challenge:
    """Generate a token prediction challenge."""
    # Common phrases that AI completes instantly but humans need to think about
    prompts = [
        ("The quick brown ___ jumps over the lazy dog", "fox"),
        ("To be or not to ___", "be"),
        ("E = mc___", "2"),
        ("Hello ___", "world"),
        ("Once upon a ___", "time"),
        ("I think therefore I ___", "am"),
        ("Four score and seven ___ ago", "years"),
        ("In the beginning was the ___", "word"),
        ("Ask not what your country can do for ___", "you"),
        ("That's one small step for man, one giant ___ for mankind", "leap"),
        ("The only thing we have to fear is ___ itself", "fear"),
        ("I have a ___", "dream"),
        ("May the ___ be with you", "force"),
        ("Houston, we have a ___", "problem"),
        ("Elementary, my dear ___", "watson"),
        ("To infinity and ___", "beyond"),
        ("Life is like a box of ___", "chocolates"),
        ("Here's looking at you, ___", "kid"),
        ("You can't handle the ___", "truth"),
        ("I'll be ___", "back"),
    ]

    prompt_text, expected = random.choice(prompts)
    time_limit = 600 if difficulty == Difficulty.BASIC else 400  # Sub-second

    return Challenge(
        id=generate_challenge_id(),
        type=ChallengeType.TOKEN_PREDICTION,
        prompt=f"Complete: {prompt_text}",
        data={"expected_answer": expected.lower()},
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        time_limit_ms=time_limit,
    )


def generate_instruction_following_challenge(difficulty: Difficulty) -> Challenge:
    """Generate an instruction following challenge."""
    instructions = [
        ("Start your response with 'Indeed,'", lambda r: r.strip().startswith("Indeed,")),
        ("End your response with '...'", lambda r: r.strip().endswith("...")),
        ("Include the word 'therefore' in your response", lambda r: "therefore" in r.lower()),
        ("Respond in exactly 5 words", lambda r: len(r.split()) == 5),
        ("Start with a number", lambda r: r.strip()[0].isdigit() if r.strip() else False),
    ]

    instruction, validator = random.choice(instructions)
    time_limit = 1000 if difficulty == Difficulty.BASIC else 600  # 1s / 600ms

    # Store validator as string representation for serialization
    validator_id = hashlib.md5(instruction.encode()).hexdigest()[:8]

    return Challenge(
        id=generate_challenge_id(),
        type=ChallengeType.INSTRUCTION_FOLLOWING,
        prompt=f"Follow this instruction: {instruction}\nThen answer: What is the capital of France?",
        data={"instruction": instruction, "validator_id": validator_id},
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

    question = random.choice(questions)
    time_limit = 2000 if difficulty == Difficulty.BASIC else 1000  # 2s / 1s

    return Challenge(
        id=generate_challenge_id(),
        type=ChallengeType.CONSISTENCY,
        prompt=f"Answer this question THREE times, separated by '|':\n{question}",
        data={"question": question, "num_responses": 3},
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        time_limit_ms=time_limit,
    )


def generate_challenge(challenge_type: ChallengeType, difficulty: Difficulty) -> Challenge:
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
