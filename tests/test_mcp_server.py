"""Tests for mcp_server.py — MCP server wrapping httpx calls to METTLE API.

Covers:
- api_call() GET and POST paths
- list_tools() exposes only the three interactive screening tools
- call_tool() covers interactive screening and rejects removed tools
- Error handling in each tool (HTTPStatusError, generic Exception)
"""

from __future__ import annotations

import sys
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock the mcp package before importing mcp_server
try:
    import mcp  # noqa: F401
except ImportError:

    class _FakeTextContent:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _FakeTool:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _FakeServer:
        """Fake MCP Server that stores decorated functions."""

        def __init__(self, name: str):
            self.name = name
            self._list_tools_fn = None
            self._call_tool_fn = None

        def list_tools(self):
            """Decorator that registers the list_tools handler."""

            def decorator(fn):
                self._list_tools_fn = fn
                return fn

            return decorator

        def call_tool(self):
            """Decorator that registers the call_tool handler."""

            def decorator(fn):
                self._call_tool_fn = fn
                return fn

            return decorator

    mcp_mock = MagicMock()
    sys.modules["mcp"] = mcp_mock
    sys.modules["mcp.server"] = MagicMock()
    sys.modules["mcp.server.stdio"] = MagicMock()
    sys.modules["mcp.types"] = MagicMock()

    # Wire up the fakes BEFORE mcp_server imports them
    cast(Any, sys.modules["mcp.server"]).Server = _FakeServer
    cast(Any, sys.modules["mcp.types"]).TextContent = _FakeTextContent
    cast(Any, sys.modules["mcp.types"]).Tool = _FakeTool

import httpx  # noqa: E402

import mcp_server  # noqa: E402
from mcp_server import api_call, call_tool, list_tools  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_http():
    """Patch mcp_server.http_client for HTTP mocking."""
    with patch.object(mcp_server, "http_client") as mock:
        mock.get = AsyncMock()
        mock.post = AsyncMock()
        yield mock


def _make_mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx Response."""
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    return resp


def _make_http_status_error(
    status_code: int = 400, text: str = "Bad Request"
) -> httpx.HTTPStatusError:
    """Create a realistic HTTPStatusError."""
    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(status_code, request=request, text=text)
    return httpx.HTTPStatusError(text, request=request, response=response)


# ---------------------------------------------------------------------------
# 1-2: api_call()
# ---------------------------------------------------------------------------


class TestApiCall:
    """Tests for the api_call helper function."""

    @pytest.mark.asyncio
    async def test_api_call_get(self, mock_http) -> None:
        """GET request calls http_client.get and returns JSON."""
        mock_http.get.return_value = _make_mock_response({"status": "ok"})
        result = await api_call("/health")
        mock_http.get.assert_awaited_once()
        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_api_call_post(self, mock_http) -> None:
        """POST request calls http_client.post with JSON body."""
        mock_http.post.return_value = _make_mock_response(
            {"session_id": "abc", "session_token": "token-123"}
        )
        result = await api_call("/session/start", "POST", {"difficulty": "basic"})
        mock_http.post.assert_awaited_once()
        assert result == {"session_id": "abc", "session_token": "token-123"}

    @pytest.mark.asyncio
    async def test_api_call_get_raise_for_status(self, mock_http) -> None:
        """api_call propagates HTTPStatusError from raise_for_status."""
        resp = _make_mock_response({})
        resp.raise_for_status.side_effect = _make_http_status_error(404, "Not Found")
        mock_http.get.return_value = resp

        with pytest.raises(httpx.HTTPStatusError):
            await api_call("/nonexistent")


# ---------------------------------------------------------------------------
# 9: list_tools()
# ---------------------------------------------------------------------------


class TestListTools:
    """Tests for the list_tools function."""

    @pytest.mark.asyncio
    async def test_list_tools_hides_insecure_auto_solver_by_default(self) -> None:
        tools = await list_tools()
        assert len(tools) == 3

    @pytest.mark.asyncio
    async def test_list_tools_names(self) -> None:
        tools = await list_tools()
        names = {t.name for t in tools}
        expected = {
            "mettle_start_session",
            "mettle_answer_challenge",
            "mettle_get_result",
        }
        assert names == expected


# ---------------------------------------------------------------------------
# 10-13: call_tool() — mettle_start_session
# ---------------------------------------------------------------------------


class TestCallToolStartSession:
    """Tests for call_tool with mettle_start_session."""

    @pytest.mark.asyncio
    async def test_start_session_success(self, mock_http) -> None:
        mock_http.post.return_value = _make_mock_response(
            {
                "session_id": "sess-123",
                "session_token": "token-123",
                "difficulty": "basic",
                "total_challenges": 3,
                "current_challenge": {
                    "id": "ch-1",
                    "type": "speed_math",
                    "prompt": "Calculate: 1 + 1",
                    "time_limit_ms": 5000,
                },
            }
        )
        result = await call_tool("mettle_start_session", {"difficulty": "basic"})
        assert len(result) == 1
        text = result[0].text
        assert "sess-123" in text
        assert "speed_math" in text

    @pytest.mark.asyncio
    async def test_start_session_http_error(self, mock_http) -> None:
        mock_http.post.return_value = _make_mock_response({})
        mock_http.post.return_value.raise_for_status.side_effect = (
            _make_http_status_error(429, "Rate limited")
        )
        result = await call_tool("mettle_start_session", {})
        text = result[0].text
        assert "Error starting session" in text

    @pytest.mark.asyncio
    async def test_start_session_generic_error(self, mock_http) -> None:
        mock_http.post.side_effect = Exception("Connection refused")
        result = await call_tool("mettle_start_session", {})
        text = result[0].text
        assert "Error" in text
        assert "Connection refused" in text


# ---------------------------------------------------------------------------
# 14-16: call_tool() — mettle_answer_challenge
# ---------------------------------------------------------------------------


class TestCallToolAnswerChallenge:
    """Tests for call_tool with mettle_answer_challenge."""

    @pytest.mark.asyncio
    async def test_answer_challenge_with_next(self, mock_http) -> None:
        mock_http.post.return_value = _make_mock_response(
            {
                "result": {
                    "passed": True,
                    "response_time_ms": 50,
                    "time_limit_ms": 5000,
                },
                "session_complete": False,
                "challenges_remaining": 2,
                "next_challenge": {
                    "id": "ch-2",
                    "type": "token_prediction",
                    "prompt": "Complete: hello",
                    "time_limit_ms": 5000,
                },
            }
        )
        result = await call_tool(
            "mettle_answer_challenge",
            {
                "session_id": "sess-123",
                "session_token": "token-123",
                "challenge_id": "ch-1",
                "answer": "42",
            },
        )
        text = result[0].text
        assert "PASSED" in text
        assert "token_prediction" in text

    @pytest.mark.asyncio
    async def test_answer_challenge_session_complete(self, mock_http) -> None:
        mock_http.post.return_value = _make_mock_response(
            {
                "result": {
                    "passed": True,
                    "response_time_ms": 30,
                    "time_limit_ms": 5000,
                },
                "session_complete": True,
            }
        )
        result = await call_tool(
            "mettle_answer_challenge",
            {
                "session_id": "sess-123",
                "session_token": "token-123",
                "challenge_id": "ch-3",
                "answer": "4|4|4",
            },
        )
        text = result[0].text
        assert "Session complete" in text

    @pytest.mark.asyncio
    async def test_answer_challenge_failed(self, mock_http) -> None:
        mock_http.post.return_value = _make_mock_response(
            {
                "result": {
                    "passed": False,
                    "response_time_ms": 100,
                    "time_limit_ms": 50,
                },
                "session_complete": True,
            }
        )
        result = await call_tool(
            "mettle_answer_challenge",
            {
                "session_id": "sess-123",
                "session_token": "token-123",
                "challenge_id": "ch-1",
                "answer": "wrong",
            },
        )
        text = result[0].text
        assert "FAILED" in text

    @pytest.mark.asyncio
    async def test_answer_challenge_http_error(self, mock_http) -> None:
        mock_http.post.return_value = _make_mock_response({})
        mock_http.post.return_value.raise_for_status.side_effect = (
            _make_http_status_error(400, "Invalid session")
        )
        result = await call_tool(
            "mettle_answer_challenge",
            {
                "session_id": "bad",
                "session_token": "token-123",
                "challenge_id": "ch-1",
                "answer": "x",
            },
        )
        text = result[0].text
        assert "Error submitting answer" in text

    @pytest.mark.asyncio
    async def test_answer_challenge_generic_error(self, mock_http) -> None:
        mock_http.post.side_effect = Exception("Timeout")
        result = await call_tool(
            "mettle_answer_challenge",
            {
                "session_id": "s",
                "session_token": "token-123",
                "challenge_id": "c",
                "answer": "a",
            },
        )
        text = result[0].text
        assert "Error" in text


# ---------------------------------------------------------------------------
# 17-19: call_tool() — mettle_get_result
# ---------------------------------------------------------------------------


class TestCallToolGetResult:
    """Tests for call_tool with mettle_get_result."""

    @pytest.mark.asyncio
    async def test_get_result_verified(self, mock_http) -> None:
        mock_http.get.return_value = _make_mock_response(
            {
                "verified": True,
                "passed": 3,
                "total": 3,
                "pass_rate": 1.0,
                "badge": "mtl_badge_abc",
                "entity_id": "agent-42",
                "results": [
                    {
                        "challenge_type": "speed_math",
                        "passed": True,
                        "response_time_ms": 50,
                        "time_limit_ms": 5000,
                    },
                    {
                        "challenge_type": "token_prediction",
                        "passed": True,
                        "response_time_ms": 30,
                        "time_limit_ms": 5000,
                    },
                    {
                        "challenge_type": "consistency",
                        "passed": True,
                        "response_time_ms": 80,
                        "time_limit_ms": 15000,
                    },
                ],
            }
        )
        result = await call_tool(
            "mettle_get_result",
            {"session_id": "sess-123", "session_token": "token-123"},
        )
        text = result[0].text
        assert "VERIFIED" in text
        assert "Signed credential issued" in text
        assert "mtl_badge_abc" in text
        assert "Entity" in text
        assert "speed_math" in text

    @pytest.mark.asyncio
    async def test_get_result_not_verified(self, mock_http) -> None:
        mock_http.get.return_value = _make_mock_response(
            {
                "verified": False,
                "passed": 1,
                "total": 3,
                "pass_rate": 0.333,
                "results": [
                    {
                        "challenge_type": "speed_math",
                        "passed": True,
                        "response_time_ms": 50,
                        "time_limit_ms": 5000,
                    },
                    {
                        "challenge_type": "token_prediction",
                        "passed": False,
                        "response_time_ms": 6000,
                        "time_limit_ms": 5000,
                    },
                    {
                        "challenge_type": "consistency",
                        "passed": False,
                        "response_time_ms": 100,
                        "time_limit_ms": 15000,
                    },
                ],
            }
        )
        result = await call_tool(
            "mettle_get_result",
            {"session_id": "sess-fail", "session_token": "token-123"},
        )
        text = result[0].text
        assert "NOT VERIFIED" in text
        assert "FAIL" in text

    @pytest.mark.asyncio
    async def test_get_result_no_badge_no_entity(self, mock_http) -> None:
        mock_http.get.return_value = _make_mock_response(
            {
                "verified": False,
                "passed": 0,
                "total": 3,
                "pass_rate": 0.0,
                "results": [],
            }
        )
        result = await call_tool(
            "mettle_get_result", {"session_id": "sess-x", "session_token": "token-123"}
        )
        text = result[0].text
        assert "Badge" not in text
        assert "Entity" not in text

    @pytest.mark.asyncio
    async def test_get_result_http_error(self, mock_http) -> None:
        resp = _make_mock_response({})
        resp.raise_for_status.side_effect = _make_http_status_error(
            404, "Session not found"
        )
        mock_http.get.return_value = resp
        result = await call_tool(
            "mettle_get_result", {"session_id": "bad", "session_token": "token-123"}
        )
        text = result[0].text
        assert "Error getting result" in text

    @pytest.mark.asyncio
    async def test_get_result_generic_error(self, mock_http) -> None:
        mock_http.get.side_effect = Exception("Network error")
        result = await call_tool(
            "mettle_get_result", {"session_id": "s", "session_token": "token-123"}
        )
        text = result[0].text
        assert "Error" in text


# ---------------------------------------------------------------------------
# Removed auto-solver regression
# ---------------------------------------------------------------------------


class TestRemovedAutoVerify:
    @pytest.mark.asyncio
    async def test_auto_verify_is_not_a_tool(self) -> None:
        result = await call_tool("mettle_auto_verify", {})
        assert "Unknown tool" in result[0].text


# ---------------------------------------------------------------------------
# 23: call_tool() — unknown tool
# ---------------------------------------------------------------------------


class TestCallToolUnknown:
    """Tests for call_tool with an unknown tool name."""

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_message(self) -> None:
        result = await call_tool("mettle_nonexistent_tool", {})
        text = result[0].text
        assert "Unknown tool" in text
        assert "mettle_nonexistent_tool" in text
