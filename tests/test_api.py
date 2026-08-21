"""Tests for METTLE API endpoints."""

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main as main_module
from main import MAX_REQUEST_BODY_BYTES, RequestBodyLimitMiddleware, app


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
        assert "expected_answer" not in challenge["data"], (
            "SECURITY: expected_answer exposed!"
        )
        assert "chain" not in challenge["data"], "SECURITY: chain exposed!"

    def test_answer_response_no_expected_answer(self, client):
        """Answers must not be in next challenge response."""
        # Start session
        start = client.post("/api/session/start", json={"difficulty": "basic"})
        session_id = start.json()["session_id"]
        session_token = start.json()["session_token"]
        challenge = start.json()["current_challenge"]

        # Submit any answer
        response = client.post(
            "/api/session/answer",
            json={
                "session_id": session_id,
                "challenge_id": challenge["id"],
                "answer": "test",
            },
            headers={"X-Session-Token": session_token},
        )
        assert response.status_code == 200
        next_challenge = response.json().get("next_challenge")
        assert next_challenge is not None
        assert "expected_answer" not in next_challenge["data"], (
            "SECURITY: expected_answer exposed!"
        )
        assert "chain" not in next_challenge["data"], "SECURITY: chain exposed!"


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_reports_degraded_when_v2_store_is_unavailable(self, client):
        """A live legacy surface must not hide an unavailable Redis-backed API."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["v2_session_store"] == "unavailable"
        assert "timestamp" in data

    def test_health_reports_healthy_when_v2_store_is_ready(self, client, monkeypatch):
        redis = MagicMock()
        redis.ping = AsyncMock(return_value=True)
        monkeypatch.setattr(app.state, "redis", redis, raising=False)

        response = client.get("/api/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["components"]["v2_session_store"] == "ready"
        redis.ping.assert_awaited_once()

    def test_health_degrades_when_an_existing_redis_connection_stops_responding(
        self, client, monkeypatch
    ):
        redis = MagicMock()
        redis.ping = AsyncMock(side_effect=TimeoutError("timed out"))
        monkeypatch.setattr(app.state, "redis", redis, raising=False)

        response = client.get("/api/health")

        assert response.status_code == 200
        assert response.json()["status"] == "degraded"
        assert response.json()["components"]["v2_session_store"] == "unavailable"


class TestRequestBodyLimit:
    """The global body limit is streaming, exact, and middleware-complete."""

    def test_declared_oversize_is_rejected_before_generation_with_headers(self, client):
        with patch("main.generate_challenge_set") as generate:
            response = client.post(
                "/api/session/start",
                content=b"x" * (MAX_REQUEST_BODY_BYTES + 1),
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 413
        assert response.json() == {"detail": "Request body too large"}
        generate.assert_not_called()
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-Request-ID"]

    @pytest.mark.asyncio
    async def test_chunked_body_without_content_length_is_bounded(self):
        downstream_called = False

        async def downstream(_scope, _receive, _send):
            nonlocal downstream_called
            downstream_called = True

        middleware = RequestBodyLimitMiddleware(
            downstream,
            max_body_bytes=MAX_REQUEST_BODY_BYTES,
        )
        messages = iter(
            [
                {
                    "type": "http.request",
                    "body": b"a" * (MAX_REQUEST_BODY_BYTES // 2),
                    "more_body": True,
                },
                {
                    "type": "http.request",
                    "body": b"b" * (MAX_REQUEST_BODY_BYTES // 2 + 1),
                    "more_body": False,
                },
            ]
        )
        sent = []

        async def receive():
            return next(messages)

        async def send(message):
            sent.append(message)

        await middleware({"type": "http", "headers": []}, receive, send)

        assert downstream_called is False
        assert sent[0]["status"] == 413

    @pytest.mark.asyncio
    async def test_exact_streaming_boundary_is_replayed_unchanged(self):
        received_body = bytearray()

        async def downstream(_scope, receive, send):
            while True:
                message = await receive()
                received_body.extend(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            await send({"type": "http.response.start", "status": 204, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        middleware = RequestBodyLimitMiddleware(
            downstream,
            max_body_bytes=MAX_REQUEST_BODY_BYTES,
        )
        chunks = [
            {
                "type": "http.request",
                "body": b"a" * (MAX_REQUEST_BODY_BYTES // 2),
                "more_body": True,
            },
            {
                "type": "http.request",
                "body": b"b" * (MAX_REQUEST_BODY_BYTES // 2),
                "more_body": False,
            },
        ]
        sent = []

        async def receive():
            return chunks.pop(0)

        async def send(message):
            sent.append(message)

        await middleware({"type": "http", "headers": []}, receive, send)

        assert len(received_body) == MAX_REQUEST_BODY_BYTES
        assert sent[0]["status"] == 204

    @pytest.mark.asyncio
    async def test_zero_byte_frame_flood_is_bounded(self):
        downstream_called = False

        async def downstream(_scope, _receive, _send):
            nonlocal downstream_called
            downstream_called = True

        middleware = RequestBodyLimitMiddleware(
            downstream,
            max_body_bytes=MAX_REQUEST_BODY_BYTES,
            max_body_frames=2,
        )
        messages = iter(
            {"type": "http.request", "body": b"", "more_body": True} for _ in range(3)
        )
        sent = []

        async def receive():
            return next(messages)

        async def send(message):
            sent.append(message)

        await middleware({"type": "http", "headers": []}, receive, send)

        assert downstream_called is False
        assert sent[0]["status"] == 400

    @pytest.mark.asyncio
    async def test_malformed_content_length_is_rejected_without_reading(self):
        async def downstream(_scope, _receive, _send):
            raise AssertionError("downstream must not run")

        async def receive():
            raise AssertionError("malformed length must be rejected before body read")

        sent = []

        async def send(message):
            sent.append(message)

        middleware = RequestBodyLimitMiddleware(
            downstream,
            max_body_bytes=MAX_REQUEST_BODY_BYTES,
        )
        await middleware(
            {"type": "http", "headers": [(b"content-length", b"1x")]},
            receive,
            send,
        )

        assert sent[0]["status"] == 400

    @pytest.mark.asyncio
    async def test_content_length_mismatch_is_rejected(self):
        downstream_called = False

        async def downstream(_scope, _receive, _send):
            nonlocal downstream_called
            downstream_called = True

        async def receive():
            return {"type": "http.request", "body": b"a", "more_body": False}

        sent = []

        async def send(message):
            sent.append(message)

        middleware = RequestBodyLimitMiddleware(
            downstream,
            max_body_bytes=MAX_REQUEST_BODY_BYTES,
        )
        await middleware(
            {"type": "http", "headers": [(b"content-length", b"2")]},
            receive,
            send,
        )

        assert downstream_called is False
        assert sent[0]["status"] == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("headers", "message"),
        [
            ("not-a-header-list", {"type": "http.request", "body": b""}),
            ([], "not-a-message-object"),
            ([], {}),
            ([], {"type": "http.request", "body": "not-bytes"}),
        ],
    )
    async def test_malformed_asgi_shapes_fail_closed(self, headers, message):
        downstream_called = False

        async def downstream(_scope, _receive, _send):
            nonlocal downstream_called
            downstream_called = True

        async def receive():
            return message

        sent = []

        async def send(item):
            sent.append(item)

        middleware = RequestBodyLimitMiddleware(
            downstream,
            max_body_bytes=MAX_REQUEST_BODY_BYTES,
        )
        await middleware(
            {"type": "http", "headers": headers},
            receive,
            send,
        )

        assert downstream_called is False
        assert sent[0]["status"] == 400


class TestAdversarialFailureBoundaries:
    def test_internal_error_is_sanitized_and_keeps_security_headers(self, client):
        with patch(
            "main.generate_challenge_set",
            side_effect=RuntimeError("sensitive backend detail"),
        ):
            response = client.post("/api/session/start", json={})

        assert response.status_code == 500
        assert response.json() == {"detail": "Internal server error"}
        assert "sensitive" not in response.text
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-Request-ID"]

    def test_capacity_is_checked_before_challenge_generation(self, client, monkeypatch):
        monkeypatch.setattr(main_module, "MAX_SESSIONS", 0)
        with patch("main.generate_challenge_set") as generate:
            response = client.post("/api/session/start", json={})

        assert response.status_code == 503
        generate.assert_not_called()

    def test_cleanup_removes_expired_owned_and_orphaned_challenges(self, client):
        created = client.post("/api/session/start", json={}).json()
        session_id = created["session_id"]
        challenge_id = created["current_challenge"]["id"]
        session = main_module.sessions[session_id]
        challenge = main_module.challenges[challenge_id][0]
        now = time.time()
        session.started_at = datetime.fromtimestamp(
            now - main_module.LEGACY_SESSION_RECOVERY_SECONDS - 1,
            tz=timezone.utc,
        )
        main_module.challenges["stale-orphan"] = (
            challenge,
            now - main_module.LEGACY_SESSION_RECOVERY_SECONDS - 1,
        )
        main_module.challenges["disarmed-orphan"] = (challenge, None)

        removed = main_module.cleanup_expired_state(now=now)

        assert removed == (1, 2)
        assert session_id not in main_module.sessions
        assert challenge_id not in main_module.challenges
        assert "stale-orphan" not in main_module.challenges
        assert "disarmed-orphan" in main_module.challenges


class TestStartSession:
    """Test session start endpoint."""

    def test_start_session_basic(self, client):
        """Test starting a basic session."""
        response = client.post("/api/session/start", json={"difficulty": "basic"})
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "session_token" in data
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
        response = client.post(
            "/api/session/start",
            json={"difficulty": "basic", "entity_id": "test-agent-001"},
        )
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


class TestLegacySessionAuthorization:
    def test_answer_requires_independent_session_token(self):
        client = TestClient(app)
        started = client.post("/api/session/start", json={}).json()
        body = {
            "session_id": started["session_id"],
            "challenge_id": started["current_challenge"]["id"],
            "answer": "test",
        }

        missing = client.post("/api/session/answer", json=body)
        wrong = client.post(
            "/api/session/answer",
            json=body,
            headers={"X-Session-Token": "wrong-token"},
        )
        correct = client.post(
            "/api/session/answer",
            json=body,
            headers={"X-Session-Token": started["session_token"]},
        )

        assert missing.status_code == 401
        assert wrong.status_code == 403
        assert correct.status_code == 200

    def test_status_and_result_require_session_token(self):
        client = TestClient(app)
        started = client.post("/api/session/start", json={}).json()
        session_id = started["session_id"]

        assert client.get(f"/api/session/{session_id}").status_code == 401
        assert client.get(f"/api/session/{session_id}/result").status_code == 401
        assert (
            client.get(
                f"/api/session/{session_id}",
                headers={"X-Session-Token": started["session_token"]},
            ).status_code
            == 200
        )


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
            "/api/session/answer",
            json={
                "session_id": session_id,
                "challenge_id": challenge["id"],
                "answer": answer,
            },
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
            json={
                "session_id": "ses_000000000000000000000000",
                "challenge_id": "mtl_000000000000000000000000",
                "answer": "test",
            },
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
            "/api/session/answer",
            json={
                "session_id": session_id,
                "challenge_id": "mtl_000000000000000000000000",
                "answer": "test",
            },
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
            "/api/session/answer",
            json={
                "session_id": session_id,
                "challenge_id": challenge["id"],
                "answer": answer,
            },
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
        start_response = client.post(
            "/api/session/start", json={"difficulty": difficulty}
        )
        start_data = start_response.json()
        session_id = start_data["session_id"]

        # Answer all challenges
        challenge = start_data["current_challenge"]
        while challenge:
            answer = TestSubmitAnswer()._solve_challenge(challenge)
            response = client.post(
                "/api/session/answer",
                json={
                    "session_id": session_id,
                    "challenge_id": challenge["id"],
                    "answer": answer,
                },
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
        start_response = client.post(
            "/api/session/start",
            json={"difficulty": "basic", "entity_id": "test-agent"},
        )
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
                json={
                    "session_id": session_id,
                    "challenge_id": challenge["id"],
                    "answer": answer,
                },
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
            "/api/session/answer",
            json={
                "session_id": session_id,
                "challenge_id": "mtl_000000000000000000000000",
                "answer": "test",
            },
        )
        assert response.status_code == 404  # Generic "not found" for security
        assert (
            "not found" in response.json()["detail"].lower()
            or "invalid" in response.json()["detail"].lower()
        )
