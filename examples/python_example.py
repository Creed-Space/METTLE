"""
METTLE Python SDK Example

Complete verification flow for AI agents.

Requirements:
    pip install httpx

Usage:
    python python_example.py
"""

import httpx
import re

# Configuration
METTLE_API = "https://mettle-api.onrender.com/api"
ENTITY_ID = "my-python-agent"


def verify_agent(entity_id: str, difficulty: str = "basic") -> dict:
    """Complete METTLE verification flow."""

    with httpx.Client(timeout=30.0) as client:
        # Step 1: Start session
        print(f"Starting verification for {entity_id}...")
        response = client.post(
            f"{METTLE_API}/session/start",
            json={"entity_id": entity_id, "difficulty": difficulty},
        )
        response.raise_for_status()
        session = response.json()

        session_id = session["session_id"]
        total = session["total_challenges"]
        print(f"Session {session_id}: {total} challenges")

        # Step 2: Answer challenges
        current_challenge = session["current_challenge"]
        challenge_num = 1

        while current_challenge:
            challenge_id = current_challenge["id"]
            challenge_type = current_challenge["type"]
            prompt = current_challenge["prompt"]

            print(f"\nChallenge {challenge_num}/{total}: {challenge_type}")
            print(f"  Prompt: {prompt[:60]}...")

            # Generate answer based on challenge type
            answer = generate_answer(challenge_type, prompt, current_challenge.get("data", {}))

            # Submit answer
            response = client.post(
                f"{METTLE_API}/session/answer",
                json={
                    "session_id": session_id,
                    "challenge_id": challenge_id,
                    "answer": answer,
                },
            )
            response.raise_for_status()
            result = response.json()

            passed = result["result"]["passed"]
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

            # Get next challenge
            current_challenge = result.get("next_challenge")
            challenge_num += 1

        # Step 3: Get final result
        response = client.get(f"{METTLE_API}/session/{session_id}/result")
        response.raise_for_status()
        final = response.json()

        print(f"\n{'=' * 40}")
        print(f"VERIFICATION {'PASSED' if final['verified'] else 'FAILED'}")
        print(f"Pass rate: {final['pass_rate'] * 100:.0f}%")
        if final.get("badge"):
            print(f"Badge: {final['badge'][:40]}...")
        if final.get("badge_info"):
            print(f"Expires: {final['badge_info']['expires_at']}")

        return final


def safe_math_eval(expr: str) -> int:
    """Safely evaluate simple math expressions (+ - * only)."""
    # Only allow digits, spaces, and basic operators
    if not re.match(r'^[\d\s\+\-\*]+$', expr):
        return 0

    # Parse and compute safely
    try:
        # Simple recursive descent for + and -
        result = 0
        current = 0
        op = '+'

        for char in expr + '+':
            if char.isdigit():
                current = current * 10 + int(char)
            elif char in '+-*':
                if op == '+':
                    result += current
                elif op == '-':
                    result -= current
                elif op == '*':
                    result *= current
                current = 0
                op = char
            elif char == ' ':
                continue

        return result
    except Exception:
        return 0


def generate_answer(challenge_type: str, prompt: str, data: dict) -> str:
    """Generate answer for a challenge. Replace with your AI's logic."""

    if challenge_type == "speed_math":
        # Parse and solve math problem
        # Example: "Calculate: 47 + 83"
        try:
            expr = prompt.split(": ")[1]
            result = safe_math_eval(expr)
            return str(result)
        except Exception:
            return "0"

    elif challenge_type == "token_prediction":
        # Complete the phrase
        # Example: "Complete: The quick brown fox jumps over the lazy ___"
        if "lazy" in prompt.lower():
            return "dog"
        elif "roses are" in prompt.lower():
            return "red"
        return "unknown"

    elif challenge_type == "instruction_following":
        instruction = data.get("instruction", "")
        if "Indeed," in instruction:
            return "Indeed, I understand the requirement."
        elif "..." in instruction:
            return "Here is my response..."
        elif "therefore" in instruction:
            return "Therefore, this follows logically."
        elif "5 words" in instruction:
            return "This has five words exactly."
        elif "number" in instruction:
            return "42 is the answer here."
        return "Response following instructions."

    elif challenge_type == "consistency":
        # Answer the same question multiple times
        num_responses = data.get("num_responses", 3)
        base = "The sky is blue"
        return " | ".join([base] * num_responses)

    elif challenge_type == "chained_reasoning":
        # Multi-step calculation
        try:
            chain = data.get("chain", [])
            result = chain[0]["value"] if chain else 0
            for step in chain[1:]:
                op = step["op"]
                val = step["value"]
                if op == "+":
                    result += val
                elif op == "-":
                    result -= val
                elif op == "*":
                    result *= val
            return str(result)
        except Exception:
            return "0"

    return "default answer"


def verify_badge(badge_token: str) -> dict:
    """Verify a METTLE badge is valid."""
    with httpx.Client() as client:
        response = client.get(f"{METTLE_API}/badge/verify/{badge_token}")
        return response.json()


if __name__ == "__main__":
    result = verify_agent(ENTITY_ID, difficulty="basic")

    if result.get("badge"):
        print("\nVerifying badge...")
        badge_check = verify_badge(result["badge"])
        print(f"Badge valid: {badge_check['valid']}")
