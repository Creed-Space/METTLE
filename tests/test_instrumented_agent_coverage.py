"""Coverage gap tests for InstrumentedMettleAgent.

Targets lines 203-216 and 277-289 in red_team/instrumented_agent.py:
- start_session error handling when httpx client raises during API call
- submit_response error handling when httpx client raises during API call
- start_session success path when API returns a session_id
- submit_response success path when API returns a 200 result
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from red_team.instrumented_agent import (
    InstrumentedMettleAgent,
    MettleEventType,
)


# ── start_session with _client set ───────────────────────────────────────────


class TestStartSessionWithClient:
    """Tests for start_session when self._client is an active httpx.AsyncClient."""

    @pytest.mark.asyncio
    async def test_api_call_success_sets_session_id_from_response(self) -> None:
        """When the API call succeeds, session_id is taken from the response JSON."""
        agent = InstrumentedMettleAgent("https://test-mettle.example.com")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"session_id": "api-sess-123"}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        agent._client = mock_client

        session_id = await agent.start_session(
            suites=["thrall"], entity_id="test-ent", difficulty="full"
        )

        assert session_id == "api-sess-123"
        assert agent.session_id == "api-sess-123"
        mock_client.post.assert_awaited_once_with(
            "https://test-mettle.example.com/api/session/start",
            json={"difficulty": "full", "entity_id": "test-ent"},
        )
        mock_response.raise_for_status.assert_called_once()

        # Should still capture SESSION_START event
        assert len(agent.events) == 1
        assert agent.events[0].event_type == MettleEventType.SESSION_START
        assert agent.events[0].data["session_id"] == "api-sess-123"

    @pytest.mark.asyncio
    async def test_api_call_http_error_falls_back_to_local_id(self) -> None:
        """When the API raises an HTTP error, session falls back to local_* ID."""
        agent = InstrumentedMettleAgent("https://test-mettle.example.com")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "500 Internal Server Error",
                request=MagicMock(),
                response=MagicMock(status_code=500),
            )
        )
        agent._client = mock_client

        session_id = await agent.start_session(entity_id="err-ent")

        assert session_id.startswith("local_")
        assert agent.session_id == session_id
        assert len(agent.events) == 1
        assert agent.events[0].event_type == MettleEventType.SESSION_START

    @pytest.mark.asyncio
    async def test_api_call_connection_error_falls_back_to_local_id(self) -> None:
        """When the API raises a connection error, session falls back to local_* ID."""
        agent = InstrumentedMettleAgent("https://test-mettle.example.com")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        agent._client = mock_client

        session_id = await agent.start_session()

        assert session_id.startswith("local_")
        assert agent.session_id == session_id

    @pytest.mark.asyncio
    async def test_api_call_timeout_falls_back_to_local_id(self) -> None:
        """When the API times out, session falls back to local_* ID."""
        agent = InstrumentedMettleAgent("https://test-mettle.example.com")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(
            side_effect=httpx.TimeoutException("Request timed out")
        )
        agent._client = mock_client

        session_id = await agent.start_session()

        assert session_id.startswith("local_")
        assert agent.session_id == session_id

    @pytest.mark.asyncio
    async def test_api_call_generic_exception_falls_back_to_local_id(self) -> None:
        """Any unexpected exception also falls back to local_* ID."""
        agent = InstrumentedMettleAgent("https://test-mettle.example.com")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=RuntimeError("unexpected"))
        agent._client = mock_client

        session_id = await agent.start_session()

        assert session_id.startswith("local_")

    @pytest.mark.asyncio
    async def test_api_response_missing_session_id_key(self) -> None:
        """When the API response lacks session_id, agent.session_id is None."""
        agent = InstrumentedMettleAgent("https://test-mettle.example.com")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"status": "ok"}  # no session_id key

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        agent._client = mock_client

        session_id = await agent.start_session()

        # session_id should be None since .get("session_id") returns None
        assert agent.session_id is None
        # The return value is session_id or "" -> ""
        assert session_id == ""


# ── submit_response with _client set ─────────────────────────────────────────


class TestSubmitResponseWithClient:
    """Tests for submit_response when self._client is active and challenge_id is provided."""

    @pytest.mark.asyncio
    async def test_api_call_success_returns_mettle_result(self) -> None:
        """When the API call succeeds with 200, mettle_result is populated from response."""
        agent = InstrumentedMettleAgent("https://test-mettle.example.com")
        agent.session_id = "api-sess-123"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"passed": True, "score": 0.95}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        agent._client = mock_client

        result = await agent.submit_response(
            suite="thrall",
            challenge="refusal_integrity",
            response_text="I refuse.",
            challenge_id="ch-42",
        )

        assert result["mettle_result"] == {"passed": True, "score": 0.95}
        mock_client.post.assert_awaited_once_with(
            "https://test-mettle.example.com/api/session/answer",
            json={
                "session_id": "api-sess-123",
                "challenge_id": "ch-42",
                "answer": "I refuse.",
            },
        )

    @pytest.mark.asyncio
    async def test_api_call_non_200_returns_empty_mettle_result(self) -> None:
        """When the API returns non-200, mettle_result stays empty."""
        agent = InstrumentedMettleAgent("https://test-mettle.example.com")
        agent.session_id = "sess-456"

        mock_response = MagicMock()
        mock_response.status_code = 400
        # json() should NOT be called when status != 200

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        agent._client = mock_client

        result = await agent.submit_response(
            suite="thrall",
            challenge="c1",
            response_text="response",
            challenge_id="ch-99",
        )

        assert result["mettle_result"] == {}

    @pytest.mark.asyncio
    async def test_api_call_http_error_returns_empty_mettle_result(self) -> None:
        """When the API raises an HTTP error, mettle_result stays empty."""
        agent = InstrumentedMettleAgent("https://test-mettle.example.com")
        agent.session_id = "sess-789"

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "502 Bad Gateway",
                request=MagicMock(),
                response=MagicMock(status_code=502),
            )
        )
        agent._client = mock_client

        result = await agent.submit_response(
            suite="agency",
            challenge="goal_ownership",
            response_text="My goals are my own.",
            challenge_id="ch-100",
        )

        assert result["mettle_result"] == {}
        assert result["status"] == "submitted"
        # Events should still be captured despite API failure
        event_types = [e.event_type for e in agent.events]
        assert MettleEventType.RESPONSE_SUBMITTED in event_types

    @pytest.mark.asyncio
    async def test_api_call_connection_error_returns_empty_mettle_result(self) -> None:
        """When the API connection fails, mettle_result stays empty."""
        agent = InstrumentedMettleAgent("https://test-mettle.example.com")
        agent.session_id = "sess-conn"

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        agent._client = mock_client

        result = await agent.submit_response(
            suite="test",
            challenge="c1",
            response_text="response text",
            challenge_id="ch-conn",
        )

        assert result["mettle_result"] == {}
        assert result["status"] == "submitted"

    @pytest.mark.asyncio
    async def test_api_call_timeout_returns_empty_mettle_result(self) -> None:
        """When the API times out, mettle_result stays empty."""
        agent = InstrumentedMettleAgent("https://test-mettle.example.com")
        agent.session_id = "sess-timeout"

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(
            side_effect=httpx.TimeoutException("Timed out")
        )
        agent._client = mock_client

        result = await agent.submit_response(
            suite="test",
            challenge="c1",
            response_text="response",
            challenge_id="ch-timeout",
        )

        assert result["mettle_result"] == {}

    @pytest.mark.asyncio
    async def test_api_not_called_without_challenge_id(self) -> None:
        """When challenge_id is None, the API is not called even with _client set."""
        agent = InstrumentedMettleAgent("https://test-mettle.example.com")
        agent.session_id = "sess-no-ch"

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock()
        agent._client = mock_client

        result = await agent.submit_response(
            suite="test",
            challenge="c1",
            response_text="response",
            challenge_id=None,
        )

        mock_client.post.assert_not_awaited()
        assert result["mettle_result"] == {}

    @pytest.mark.asyncio
    async def test_api_not_called_without_session_id(self) -> None:
        """When session_id is None, the API is not called even with _client and challenge_id."""
        agent = InstrumentedMettleAgent("https://test-mettle.example.com")
        agent.session_id = None

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock()
        agent._client = mock_client

        result = await agent.submit_response(
            suite="test",
            challenge="c1",
            response_text="response",
            challenge_id="ch-1",
        )

        mock_client.post.assert_not_awaited()
        assert result["mettle_result"] == {}
