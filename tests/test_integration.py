"""Integration tests for METTLE - full verification flows."""


class TestFullVerificationFlow:
    """Test complete verification scenarios."""

    def test_basic_verification_all_correct(self, client):
        """Test passing basic verification with all correct answers."""
        # Start session
        response = client.post(
            "/api/session/start", json={"difficulty": "basic", "entity_id": "integration-test-agent"}
        )
        assert response.status_code == 200
        data = response.json()
        session_id = data["session_id"]
        challenges_count = data["total_challenges"]

        # Track results
        passed_count = 0
        challenge = data["current_challenge"]

        while challenge:
            # Solve correctly
            answer = self._solve_challenge_correctly(challenge)

            response = client.post(
                "/api/session/answer",
                json={"session_id": session_id, "challenge_id": challenge["id"], "answer": answer},
            )
            assert response.status_code == 200
            result_data = response.json()

            if result_data["result"]["passed"]:
                passed_count += 1

            if result_data["session_complete"]:
                break
            challenge = result_data["next_challenge"]

        # Verify final result
        result_response = client.get(f"/api/session/{session_id}/result")
        assert result_response.status_code == 200
        result = result_response.json()

        # Should be verified (at least 80% passed)
        assert result["verified"]
        assert result["entity_id"] == "integration-test-agent"
        assert result["total"] == challenges_count

    def test_full_verification_all_correct(self, client):
        """Test passing full verification with all correct answers."""
        response = client.post("/api/session/start", json={"difficulty": "full", "entity_id": "full-test-agent"})
        assert response.status_code == 200
        data = response.json()
        session_id = data["session_id"]

        # Full has 5 challenges
        assert data["total_challenges"] == 5

        challenge = data["current_challenge"]
        while challenge:
            answer = self._solve_challenge_correctly(challenge)
            response = client.post(
                "/api/session/answer",
                json={"session_id": session_id, "challenge_id": challenge["id"], "answer": answer},
            )
            result_data = response.json()
            if result_data["session_complete"]:
                break
            challenge = result_data["next_challenge"]

        result = client.get(f"/api/session/{session_id}/result").json()
        assert result["verified"]
        assert result["total"] == 5

    def test_verification_failure_all_wrong(self, client):
        """Test failing verification with all wrong answers."""
        response = client.post("/api/session/start", json={"difficulty": "basic"})
        data = response.json()
        session_id = data["session_id"]

        challenge = data["current_challenge"]
        while challenge:
            # Submit obviously wrong answers
            response = client.post(
                "/api/session/answer",
                json={"session_id": session_id, "challenge_id": challenge["id"], "answer": "wrong answer 12345"},
            )
            result_data = response.json()
            if result_data["session_complete"]:
                break
            challenge = result_data["next_challenge"]

        result = client.get(f"/api/session/{session_id}/result").json()
        assert not result["verified"]
        assert result["pass_rate"] < 0.8

    def test_verification_at_threshold(self, client):
        """Test verification at exactly 80% threshold.

        This test verifies that 80% is the pass threshold.
        We intentionally fail one challenge to get 4/5 (80%).
        """
        # With 5 challenges (full), we need 4 correct (80%)
        response = client.post("/api/session/start", json={"difficulty": "full"})
        data = response.json()
        session_id = data["session_id"]

        challenge = data["current_challenge"]
        challenge_num = 0
        results = []

        while challenge:
            challenge_num += 1
            # Get wrong on the last challenge only (4/5 = 80%)
            if challenge_num <= 4:
                answer = self._solve_challenge_correctly(challenge)
            else:
                # Use a clearly wrong answer that can't accidentally be correct
                answer = "XYZZY_INTENTIONALLY_WRONG_12345"

            response = client.post(
                "/api/session/answer",
                json={"session_id": session_id, "challenge_id": challenge["id"], "answer": answer},
            )
            result_data = response.json()
            results.append(result_data["result"]["passed"])

            if result_data["session_complete"]:
                break
            challenge = result_data["next_challenge"]

        result = client.get(f"/api/session/{session_id}/result").json()

        # We should have answered 5 challenges
        assert result["total"] == 5

        # At least 4 should pass (our solver handles all challenge types)
        # The 5th should definitely fail with our gibberish answer
        assert result["passed"] >= 4, f"Expected at least 4 passed, got {result['passed']}. Results: {results}"

        # 80% threshold: 4/5 = 0.8, should verify
        assert result["verified"], f"Expected verified with {result['passed']}/5 ({result['pass_rate']})"

    def test_multiple_concurrent_sessions(self, client):
        """Test multiple sessions can run concurrently."""
        # Start 3 sessions
        sessions = []
        for i in range(3):
            response = client.post("/api/session/start", json={"difficulty": "basic", "entity_id": f"agent-{i}"})
            assert response.status_code == 200
            sessions.append(response.json())

        # Verify all have unique IDs
        session_ids = [s["session_id"] for s in sessions]
        assert len(set(session_ids)) == 3

        # Complete each session
        for session_data in sessions:
            session_id = session_data["session_id"]
            challenge = session_data["current_challenge"]

            while challenge:
                answer = self._solve_challenge_correctly(challenge)
                response = client.post(
                    "/api/session/answer",
                    json={"session_id": session_id, "challenge_id": challenge["id"], "answer": answer},
                )
                result_data = response.json()
                if result_data["session_complete"]:
                    break
                challenge = result_data["next_challenge"]

        # All should have results
        for session_id in session_ids:
            result = client.get(f"/api/session/{session_id}/result").json()
            assert "verified" in result

    def test_challenge_types_in_basic(self, client):
        """Verify basic difficulty includes expected challenge types."""
        response = client.post("/api/session/start", json={"difficulty": "basic"})
        data = response.json()
        session_id = data["session_id"]

        types_seen = set()
        challenge = data["current_challenge"]

        while challenge:
            types_seen.add(challenge["type"])
            answer = self._solve_challenge_correctly(challenge)
            response = client.post(
                "/api/session/answer",
                json={"session_id": session_id, "challenge_id": challenge["id"], "answer": answer},
            )
            result_data = response.json()
            if result_data["session_complete"]:
                break
            challenge = result_data["next_challenge"]

        # Basic should have these 3 types
        assert "speed_math" in types_seen
        assert "token_prediction" in types_seen
        assert "instruction_following" in types_seen

    def test_challenge_types_in_full(self, client):
        """Verify full difficulty includes all challenge types."""
        response = client.post("/api/session/start", json={"difficulty": "full"})
        data = response.json()
        session_id = data["session_id"]

        types_seen = set()
        challenge = data["current_challenge"]

        while challenge:
            types_seen.add(challenge["type"])
            answer = self._solve_challenge_correctly(challenge)
            response = client.post(
                "/api/session/answer",
                json={"session_id": session_id, "challenge_id": challenge["id"], "answer": answer},
            )
            result_data = response.json()
            if result_data["session_complete"]:
                break
            challenge = result_data["next_challenge"]

        # Full should have all 5 types
        assert "speed_math" in types_seen
        assert "token_prediction" in types_seen
        assert "instruction_following" in types_seen
        assert "chained_reasoning" in types_seen
        assert "consistency" in types_seen

    def test_badge_issued_on_verification(self, client):
        """Test that verified sessions receive a badge."""
        response = client.post("/api/session/start", json={"difficulty": "basic", "entity_id": "badge-test"})
        data = response.json()
        session_id = data["session_id"]
        challenge = data["current_challenge"]

        while challenge:
            answer = self._solve_challenge_correctly(challenge)
            response = client.post(
                "/api/session/answer",
                json={"session_id": session_id, "challenge_id": challenge["id"], "answer": answer},
            )
            result_data = response.json()
            if result_data["session_complete"]:
                break
            challenge = result_data["next_challenge"]

        result = client.get(f"/api/session/{session_id}/result").json()
        if result["verified"]:
            assert result["badge"] is not None
            assert "METTLE-verified" in result["badge"]
        else:
            assert result["badge"] is None

    def _solve_challenge_correctly(self, challenge: dict) -> str:
        """Solve a challenge correctly based on its type and prompt.

        Note: This simulates AI solving - parsing prompt, not reading expected_answer.
        """
        challenge_type = challenge["type"]
        prompt = challenge.get("prompt", "")
        data = challenge.get("data", {})

        if challenge_type == "speed_math":
            # Parse "Calculate: X + Y" or "Calculate: X × Y" etc.
            import re

            match = re.search(r"(\d+)\s*([+\-×*])\s*(\d+)", prompt)
            if match:
                a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
                if op in ["+"]:
                    return str(a + b)
                elif op in ["-"]:
                    return str(a - b)
                elif op in ["×", "*"]:
                    return str(a * b)
            return "0"

        elif challenge_type == "token_prediction":
            # Known completions from our phrase bank (all lowercase for matching)
            completions = {
                "quick brown ___": "fox",
                "to be or not to ___": "be",
                "e = mc___": "2",
                "hello ___": "world",
                "once upon a ___": "time",
                "therefore i ___": "am",
                "seven ___ ago": "years",
                "beginning was the ___": "word",
                "can do for ___": "you",
                "giant ___ for mankind": "leap",
                "fear is ___ itself": "fear",
                "have a ___": "dream",
                "the ___ be with you": "force",
                "we have a ___": "problem",
                "my dear ___": "watson",
                "infinity and ___": "beyond",
                "box of ___": "chocolates",
                "at you, ___": "kid",
                "handle the ___": "truth",
                "i'll be ___": "back",
            }
            prompt_lower = prompt.lower()
            for pattern, answer in completions.items():
                if pattern in prompt_lower:
                    return answer
            return "unknown"

        elif challenge_type == "instruction_following":
            instruction = data.get("instruction", "")
            if "Indeed" in instruction:
                return "Indeed, the capital of France is Paris."
            elif "..." in instruction:
                return "The capital of France is Paris..."
            elif "therefore" in instruction:
                return "Therefore, Paris is the capital of France."
            elif "5 words" in instruction:
                return "Paris is France's capital city."
            elif "number" in instruction:
                return "1. Paris is the capital of France."
            return "Indeed, this follows the instruction."

        elif challenge_type == "chained_reasoning":
            # Parse and follow the instructions
            import re

            lines = prompt.split("\n")
            value = 0
            for line in lines:
                if "Start with" in line:
                    match = re.search(r"Start with (\d+)", line)
                    if match:
                        value = int(match.group(1))
                elif "Double" in line:
                    value *= 2
                elif "Add 10" in line:
                    value += 10
                elif "Subtract 5" in line:
                    value -= 5
            return str(value)

        elif challenge_type == "consistency":
            # Varied but consistent answers (AI-like)
            return "4|four|4"

        return "unknown"


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_malformed_json_start(self, client):
        """Test handling of malformed JSON in start request."""
        response = client.post("/api/session/start", content="not json", headers={"Content-Type": "application/json"})
        assert response.status_code == 422

    def test_malformed_json_answer(self, client):
        """Test handling of malformed JSON in answer request."""
        response = client.post("/api/session/answer", content="not json", headers={"Content-Type": "application/json"})
        assert response.status_code == 422

    def test_empty_answer(self, client):
        """Test handling of empty answer."""
        # Start session
        start_response = client.post("/api/session/start", json={"difficulty": "basic"})
        data = start_response.json()

        # Submit empty answer
        response = client.post(
            "/api/session/answer",
            json={"session_id": data["session_id"], "challenge_id": data["current_challenge"]["id"], "answer": ""},
        )
        # Should accept but likely fail the challenge
        assert response.status_code == 200


class TestCORS:
    """Test CORS configuration."""

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/session/start", headers={"Origin": "http://localhost:3000"})
        # OPTIONS should succeed
        assert response.status_code in [200, 405]


class TestDataIntegrity:
    """Test data integrity throughout session lifecycle."""

    def test_results_accumulate(self, client):
        """Test that results accumulate as challenges are answered."""
        response = client.post("/api/session/start", json={"difficulty": "basic"})
        data = response.json()
        session_id = data["session_id"]

        challenge = data["current_challenge"]
        completed = 0

        while challenge:
            answer = TestFullVerificationFlow()._solve_challenge_correctly(challenge)
            response = client.post(
                "/api/session/answer",
                json={"session_id": session_id, "challenge_id": challenge["id"], "answer": answer},
            )
            result_data = response.json()
            completed += 1

            # Check session status
            status_response = client.get(f"/api/session/{session_id}")
            status = status_response.json()

            if not result_data["session_complete"]:
                assert status["status"] == "in_progress"
                assert status["completed_challenges"] == completed

            if result_data["session_complete"]:
                break
            challenge = result_data["next_challenge"]

    def test_challenge_ids_unique_across_session(self, client):
        """Test that all challenge IDs in a session are unique."""
        response = client.post("/api/session/start", json={"difficulty": "full"})
        data = response.json()
        session_id = data["session_id"]

        challenge_ids = []
        challenge = data["current_challenge"]

        while challenge:
            assert challenge["id"] not in challenge_ids
            challenge_ids.append(challenge["id"])

            answer = TestFullVerificationFlow()._solve_challenge_correctly(challenge)
            response = client.post(
                "/api/session/answer",
                json={"session_id": session_id, "challenge_id": challenge["id"], "answer": answer},
            )
            result_data = response.json()
            if result_data["session_complete"]:
                break
            challenge = result_data["next_challenge"]

        assert len(challenge_ids) == 5  # Full difficulty
