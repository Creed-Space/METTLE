"""
METTLE Python SDK Example

Interactive verification flow for Becoming Minds.

Requirements:
    pip install httpx

Usage:
    python python_example.py
"""

import httpx

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
        session_headers = {"X-Session-Token": session["session_token"]}
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

            # The example deliberately does not solve issuer challenges. Supply
            # the response produced by the Becoming Mind being verified.
            answer = input("  Response: ")

            # Submit answer
            response = client.post(
                f"{METTLE_API}/session/answer",
                headers=session_headers,
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
        response = client.get(
            f"{METTLE_API}/session/{session_id}/result", headers=session_headers
        )
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


def verify_badge(badge_token: str) -> dict:
    """Verify a METTLE badge is valid."""
    with httpx.Client() as client:
        response = client.post(
            f"{METTLE_API}/badge/verify", json={"token": badge_token}
        )
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    result = verify_agent(ENTITY_ID, difficulty="basic")

    if result.get("badge"):
        print("\nVerifying badge...")
        badge_check = verify_badge(result["badge"])
        print(f"Badge valid: {badge_check['valid']}")
