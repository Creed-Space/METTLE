"""Tests for METTLE API endpoints."""


class TestRootEndpoint:
    """Test the root endpoint."""

    def test_root_returns_info(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "METTLE"
        assert "endpoints" in data

    def test_root_contains_endpoints(self, client):
        """Test root lists available endpoints."""
        response = client.get("/api")
        endpoints = response.json()["endpoints"]
        assert "POST /api/session/start" in endpoints
        assert "POST /api/session/answer" in endpoints


class TestSecurityAnswerLeakage:
    """SECURITY: Verify answers are never exposed to clients."""

    def test_start_session_no_expected_answer(self, client):
        """Answers must not be in start session response."""
        response = client.post("/api/session/start", json={"difficulty": "basic"})
        challenge = response.json()["current_challenge"]
        assert "expected_answer" not in challenge["data"], "SECURITY: expected_answer exposed!"
        assert "chain" not in challenge["data"], "SECURITY: chain exposed!"

    def test_answer_response_no_expected_answer(self, client):
        """Answers must not be in next challenge response."""
        # Start session
        start = client.post("/api/session/start", json={"difficulty": "basic"})
        session_id = start.json()["session_id"]
        challenge = start.json()["current_challenge"]

        # Submit any answer
        response = client.post(
            "/api/session/answer", json={"session_id": session_id, "challenge_id": challenge["id"], "answer": "test"}
        )
        next_challenge = response.json().get("next_challenge")
        if next_challenge:
            assert "expected_answer" not in next_challenge["data"], "SECURITY: expected_answer exposed!"
            assert "chain" not in next_challenge["data"], "SECURITY: chain exposed!"


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_returns_healthy(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestStartSession:
    """Test session start endpoint."""

    def test_start_session_basic(self, client):
        """Test starting a basic session."""
        response = client.post("/api/session/start", json={"difficulty": "basic"})
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["session_id"].startswith("ses_")
        assert data["difficulty"] == "basic"
        assert data["total_challenges"] == 3

    def test_start_session_full(self, client):
        """Test starting a full session."""
        response = client.post("/api/session/start", json={"difficulty": "full"})
        assert response.status_code == 200
        data = response.json()
        assert data["total_challenges"] == 5

    def test_start_session_with_entity_id(self, client):
        """Test starting session with entity ID."""
        response = client.post("/api/session/start", json={"difficulty": "basic", "entity_id": "test-agent-001"})
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"].startswith("ses_")

    def test_start_session_returns_challenge(self, client):
        """Test that session start returns first challenge."""
        response = client.post("/api/session/start", json={"difficulty": "basic"})
        data = response.json()
        assert "current_challenge" in data
        challenge = data["current_challenge"]
        assert "id" in challenge
        assert "type" in challenge
        assert "prompt" in challenge
        assert "time_limit_ms" in challenge

    def test_start_session_default_difficulty(self, client):
        """Test default difficulty is basic."""
        response = client.post("/api/session/start", json={})
        assert response.status_code == 200
        assert response.json()["difficulty"] == "basic"

    def test_start_session_invalid_difficulty(self, client):
        """Test invalid difficulty returns error."""
        response = client.post("/api/session/start", json={"difficulty": "impossible"})
        assert response.status_code == 422  # Validation error


class TestSubmitAnswer:
    """Test answer submission endpoint."""

    def test_submit_answer_correct(self, client):
        """Test submitting correct answer."""
        # Start session
        start_response = client.post("/api/session/start", json={"difficulty": "basic"})
        start_data = start_response.json()
        session_id = start_data["session_id"]
        challenge = start_data["current_challenge"]

        # Solve based on challenge type
        answer = self._solve_challenge(challenge)

        # Submit answer
        response = client.post(
            "/api/session/answer", json={"session_id": session_id, "challenge_id": challenge["id"], "answer": answer}
        )
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "challenges_remaining" in data

    def test_submit_answer_invalid_session(self, client):
        """Test submitting to invalid session."""
        # Use valid format IDs that don't exist
        response = client.post(
            "/api/session/answer",
            json={"session_id": "ses_000000000000000000000000", "challenge_id": "mtl_000000000000000000000000", "answer": "test"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_submit_answer_invalid_challenge(self, client):
        """Test submitting to invalid challenge."""
        # Start session
        start_response = client.post("/api/session/start", json={"difficulty": "basic"})
        session_id = start_response.json()["session_id"]

        # Use valid format ID that doesn't exist
        response = client.post(
            "/api/session/answer", json={"session_id": session_id, "challenge_id": "mtl_000000000000000000000000", "answer": "test"}
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_submit_answer_returns_next_challenge(self, client):
        """Test that answer returns next challenge."""
        # Start session
        start_response = client.post("/api/session/start", json={"difficulty": "basic"})
        start_data = start_response.json()
        session_id = start_data["session_id"]
        challenge = start_data["current_challenge"]

        answer = self._solve_challenge(challenge)

        response = client.post(
            "/api/session/answer", json={"session_id": session_id, "challenge_id": challenge["id"], "answer": answer}
        )
        data = response.json()

        # Should have next challenge if not complete
        if data["challenges_remaining"] > 0:
            assert data["next_challenge"] is not None
            assert not data["session_complete"]
        else:
            assert data["session_complete"]

    def _solve_challenge(self, challenge: dict) -> str:
        """Helper to solve a challenge for testing."""
        challenge_type = challenge["type"]
        data = challenge.get("data", {})

        if challenge_type == "speed_math":
            return str(data.get("expected_answer", 0))
        elif challenge_type == "token_prediction":
            return data.get("expected_answer", "")
        elif challenge_type == "instruction_following":
            instruction = data.get("instruction", "")
            if "Indeed" in instruction:
                return "Indeed, Paris."
            elif "..." in instruction:
                return "This is my answer..."
            elif "therefore" in instruction:
                return "Therefore, this is correct."
            elif "5 words" in instruction:
                return "Paris is the capital here."
            elif "number" in instruction:
                return "1 Paris is the capital."
            return "Indeed, this is the answer."
        elif challenge_type == "chained_reasoning":
            return str(data.get("expected_answer", 0))
        elif challenge_type == "consistency":
            return "4|4|4"
        return "unknown"


class TestGetSession:
    """Test session status endpoint."""

    def test_get_session_in_progress(self, client):
        """Test getting in-progress session."""
        # Start session
        start_response = client.post("/api/session/start", json={"difficulty": "basic"})
        session_id = start_response.json()["session_id"]

        response = client.get(f"/api/session/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "in_progress"
        assert data["completed_challenges"] == 0

    def test_get_session_not_found(self, client):
        """Test getting nonexistent session."""
        response = client.get("/session/ses_nonexistent")
        assert response.status_code == 404

    def test_get_session_completed(self, client):
        """Test getting completed session shows result."""
        # Complete a full session
        session_id = self._complete_session(client, "basic")

        response = client.get(f"/api/session/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "result" in data

    def _complete_session(self, client, difficulty: str) -> str:
        """Helper to complete a session."""
        # Start session
        start_response = client.post("/api/session/start", json={"difficulty": difficulty})
        start_data = start_response.json()
        session_id = start_data["session_id"]

        # Answer all challenges
        challenge = start_data["current_challenge"]
        while challenge:
            answer = TestSubmitAnswer()._solve_challenge(challenge)
            response = client.post(
                "/api/session/answer",
                json={"session_id": session_id, "challenge_id": challenge["id"], "answer": answer},
            )
            data = response.json()
            if data["session_complete"]:
                break
            challenge = data["next_challenge"]

        return session_id


class TestGetResult:
    """Test result endpoint."""

    def test_get_result_completed(self, client):
        """Test getting result for completed session."""
        session_id = TestGetSession()._complete_session(client, "basic")

        response = client.get(f"/api/session/{session_id}/result")
        assert response.status_code == 200
        data = response.json()
        assert "verified" in data
        assert "passed" in data
        assert "total" in data
        assert "pass_rate" in data
        assert "results" in data

    def test_get_result_not_completed(self, client):
        """Test getting result for incomplete session."""
        # Start session but don't complete
        start_response = client.post("/api/session/start", json={"difficulty": "basic"})
        session_id = start_response.json()["session_id"]

        response = client.get(f"/api/session/{session_id}/result")
        assert response.status_code == 400
        assert "not yet completed" in response.json()["detail"].lower()

    def test_get_result_not_found(self, client):
        """Test getting result for nonexistent session."""
        response = client.get("/session/ses_nonexistent/result")
        assert response.status_code == 404


class TestCompleteSessionFlow:
    """Test complete session flow end-to-end."""

    def test_complete_basic_session(self, client):
        """Test completing a basic session."""
        # Start
        start_response = client.post("/api/session/start", json={"difficulty": "basic", "entity_id": "test-agent"})
        assert start_response.status_code == 200
        start_data = start_response.json()
        session_id = start_data["session_id"]
        total = start_data["total_challenges"]
        assert total == 3

        # Answer all challenges
        challenge = start_data["current_challenge"]
        answered = 0
        while challenge:
            answer = TestSubmitAnswer()._solve_challenge(challenge)
            response = client.post(
                "/api/session/answer",
                json={"session_id": session_id, "challenge_id": challenge["id"], "answer": answer},
            )
            assert response.status_code == 200
            data = response.json()
            answered += 1

            if data["session_complete"]:
                break
            challenge = data["next_challenge"]

        assert answered == total

        # Get result
        result_response = client.get(f"/api/session/{session_id}/result")
        assert result_response.status_code == 200
        result = result_response.json()
        assert result["total"] == 3

    def test_session_cannot_answer_after_complete(self, client):
        """Test that completed session rejects new answers."""
        session_id = TestGetSession()._complete_session(client, "basic")

        # Try to submit another answer (use valid format ID)
        # SECURITY: Error message is intentionally generic to prevent session enumeration
        response = client.post(
            "/api/session/answer", json={"session_id": session_id, "challenge_id": "mtl_000000000000000000000000", "answer": "test"}
        )
        assert response.status_code == 404  # Generic "not found" for security
        assert "not found" in response.json()["detail"].lower() or "invalid" in response.json()["detail"].lower()
