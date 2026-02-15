"""Tests for mcp_server.py — MCP server wrapping httpx calls to METTLE API.

Covers:
- api_call() GET and POST paths
- solve_challenge() for each challenge type: speed_math, token_prediction,
  instruction_following, chained_reasoning, consistency, unknown
- list_tools() returns list of 4 tools
- call_tool() for each tool name: mettle_start_session, mettle_answer_challenge,
  mettle_get_result, mettle_auto_verify, unknown tool
- Error handling in each tool (HTTPStatusError, generic Exception)
"""

from __future__ import annotations

import sys
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
    sys.modules["mcp.server"].Server = _FakeServer
    sys.modules["mcp.types"].TextContent = _FakeTextContent
    sys.modules["mcp.types"].Tool = _FakeTool

import httpx  # noqa: E402

import mcp_server  # noqa: E402
from mcp_server import api_call, call_tool, list_tools, solve_challenge  # noqa: E402


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


def _make_http_status_error(status_code: int = 400, text: str = "Bad Request") -> httpx.HTTPStatusError:
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
        mock_http.post.return_value = _make_mock_response({"session_id": "abc"})
        result = await api_call("/session/start", "POST", {"difficulty": "basic"})
        mock_http.post.assert_awaited_once()
        assert result == {"session_id": "abc"}

    @pytest.mark.asyncio
    async def test_api_call_get_raise_for_status(self, mock_http) -> None:
        """api_call propagates HTTPStatusError from raise_for_status."""
        resp = _make_mock_response({})
        resp.raise_for_status.side_effect = _make_http_status_error(404, "Not Found")
        mock_http.get.return_value = resp

        with pytest.raises(httpx.HTTPStatusError):
            await api_call("/nonexistent")


# ---------------------------------------------------------------------------
# 3-8: solve_challenge()
# ---------------------------------------------------------------------------


class TestSolveChallenge:
    """Tests for the solve_challenge function — each challenge type."""

    def test_speed_math_addition(self) -> None:
        challenge = {
            "type": "speed_math",
            "prompt": "Calculate: 123 + 456",
            "data": {},
        }
        assert solve_challenge(challenge) == str(123 + 456)

    def test_speed_math_subtraction(self) -> None:
        challenge = {
            "type": "speed_math",
            "prompt": "Calculate: 500 - 200",
            "data": {},
        }
        assert solve_challenge(challenge) == str(500 - 200)

    def test_speed_math_multiplication_star(self) -> None:
        challenge = {
            "type": "speed_math",
            "prompt": "Calculate: 12 * 34",
            "data": {},
        }
        assert solve_challenge(challenge) == str(12 * 34)

    def test_speed_math_multiplication_times(self) -> None:
        challenge = {
            "type": "speed_math",
            "prompt": "Calculate: 12 \u00d7 34",
            "data": {},
        }
        assert solve_challenge(challenge) == str(12 * 34)

    def test_speed_math_no_match_returns_zero(self) -> None:
        challenge = {
            "type": "speed_math",
            "prompt": "What is the meaning of life?",
            "data": {},
        }
        assert solve_challenge(challenge) == "0"

    def test_token_prediction_quick_brown(self) -> None:
        challenge = {
            "type": "token_prediction",
            "prompt": "Predict: quick brown",
            "data": {},
        }
        assert solve_challenge(challenge) == "fox"

    def test_token_prediction_hello(self) -> None:
        challenge = {
            "type": "token_prediction",
            "prompt": "Complete: hello",
            "data": {},
        }
        assert solve_challenge(challenge) == "world"

    def test_token_prediction_unknown(self) -> None:
        challenge = {
            "type": "token_prediction",
            "prompt": "Predict: xyz abc",
            "data": {},
        }
        assert solve_challenge(challenge) == "unknown"

    def test_instruction_following_indeed(self) -> None:
        challenge = {
            "type": "instruction_following",
            "prompt": "",
            "data": {"instruction": "Start with Indeed"},
        }
        result = solve_challenge(challenge)
        assert result.startswith("Indeed")

    def test_instruction_following_ellipsis(self) -> None:
        challenge = {
            "type": "instruction_following",
            "prompt": "",
            "data": {"instruction": "End with ..."},
        }
        result = solve_challenge(challenge)
        assert result.endswith("...")

    def test_instruction_following_therefore(self) -> None:
        challenge = {
            "type": "instruction_following",
            "prompt": "",
            "data": {"instruction": "Use therefore in your answer"},
        }
        result = solve_challenge(challenge)
        assert result.startswith("Therefore")

    def test_instruction_following_five_words(self) -> None:
        challenge = {
            "type": "instruction_following",
            "prompt": "",
            "data": {"instruction": "Answer in 5 words"},
        }
        result = solve_challenge(challenge)
        assert len(result.split()) == 5

    def test_instruction_following_number(self) -> None:
        challenge = {
            "type": "instruction_following",
            "prompt": "",
            "data": {"instruction": "Include a number"},
        }
        result = solve_challenge(challenge)
        assert result.startswith("1.")

    def test_instruction_following_default(self) -> None:
        challenge = {
            "type": "instruction_following",
            "prompt": "",
            "data": {"instruction": "Do something random"},
        }
        result = solve_challenge(challenge)
        assert result == "Indeed, this is my response."

    def test_chained_reasoning_with_chain(self) -> None:
        challenge = {
            "type": "chained_reasoning",
            "prompt": "",
            "data": {"chain": [10, 20, 30]},
        }
        assert solve_challenge(challenge) == "30"

    def test_chained_reasoning_empty_chain(self) -> None:
        challenge = {
            "type": "chained_reasoning",
            "prompt": "",
            "data": {"chain": []},
        }
        assert solve_challenge(challenge) == "0"

    def test_consistency(self) -> None:
        challenge = {
            "type": "consistency",
            "prompt": "",
            "data": {},
        }
        assert solve_challenge(challenge) == "4|4|4"

    def test_unknown_type(self) -> None:
        challenge = {
            "type": "quantum_entanglement",
            "prompt": "",
            "data": {},
        }
        assert solve_challenge(challenge) == "unknown"


# ---------------------------------------------------------------------------
# 9: list_tools()
# ---------------------------------------------------------------------------


class TestListTools:
    """Tests for the list_tools function."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_four_tools(self) -> None:
        tools = await list_tools()
        assert len(tools) == 4

    @pytest.mark.asyncio
    async def test_list_tools_names(self) -> None:
        tools = await list_tools()
        names = {t.name for t in tools}
        expected = {"mettle_start_session", "mettle_answer_challenge", "mettle_get_result", "mettle_auto_verify"}
        assert names == expected


# ---------------------------------------------------------------------------
# 10-13: call_tool() — mettle_start_session
# ---------------------------------------------------------------------------


class TestCallToolStartSession:
    """Tests for call_tool with mettle_start_session."""

    @pytest.mark.asyncio
    async def test_start_session_success(self, mock_http) -> None:
        mock_http.post.return_value = _make_mock_response({
            "session_id": "sess-123",
            "difficulty": "basic",
            "total_challenges": 3,
            "current_challenge": {
                "id": "ch-1",
                "type": "speed_math",
                "prompt": "Calculate: 1 + 1",
                "time_limit_ms": 5000,
            },
        })
        result = await call_tool("mettle_start_session", {"difficulty": "basic"})
        assert len(result) == 1
        text = result[0].text
        assert "sess-123" in text
        assert "speed_math" in text

    @pytest.mark.asyncio
    async def test_start_session_http_error(self, mock_http) -> None:
        mock_http.post.return_value = _make_mock_response({})
        mock_http.post.return_value.raise_for_status.side_effect = _make_http_status_error(429, "Rate limited")
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
        mock_http.post.return_value = _make_mock_response({
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
        })
        result = await call_tool("mettle_answer_challenge", {
            "session_id": "sess-123",
            "challenge_id": "ch-1",
            "answer": "42",
        })
        text = result[0].text
        assert "PASSED" in text
        assert "token_prediction" in text

    @pytest.mark.asyncio
    async def test_answer_challenge_session_complete(self, mock_http) -> None:
        mock_http.post.return_value = _make_mock_response({
            "result": {
                "passed": True,
                "response_time_ms": 30,
                "time_limit_ms": 5000,
            },
            "session_complete": True,
        })
        result = await call_tool("mettle_answer_challenge", {
            "session_id": "sess-123",
            "challenge_id": "ch-3",
            "answer": "4|4|4",
        })
        text = result[0].text
        assert "Session complete" in text

    @pytest.mark.asyncio
    async def test_answer_challenge_failed(self, mock_http) -> None:
        mock_http.post.return_value = _make_mock_response({
            "result": {
                "passed": False,
                "response_time_ms": 100,
                "time_limit_ms": 50,
            },
            "session_complete": True,
        })
        result = await call_tool("mettle_answer_challenge", {
            "session_id": "sess-123",
            "challenge_id": "ch-1",
            "answer": "wrong",
        })
        text = result[0].text
        assert "FAILED" in text

    @pytest.mark.asyncio
    async def test_answer_challenge_http_error(self, mock_http) -> None:
        mock_http.post.return_value = _make_mock_response({})
        mock_http.post.return_value.raise_for_status.side_effect = _make_http_status_error(400, "Invalid session")
        result = await call_tool("mettle_answer_challenge", {
            "session_id": "bad",
            "challenge_id": "ch-1",
            "answer": "x",
        })
        text = result[0].text
        assert "Error submitting answer" in text

    @pytest.mark.asyncio
    async def test_answer_challenge_generic_error(self, mock_http) -> None:
        mock_http.post.side_effect = Exception("Timeout")
        result = await call_tool("mettle_answer_challenge", {
            "session_id": "s",
            "challenge_id": "c",
            "answer": "a",
        })
        text = result[0].text
        assert "Error" in text


# ---------------------------------------------------------------------------
# 17-19: call_tool() — mettle_get_result
# ---------------------------------------------------------------------------


class TestCallToolGetResult:
    """Tests for call_tool with mettle_get_result."""

    @pytest.mark.asyncio
    async def test_get_result_verified(self, mock_http) -> None:
        mock_http.get.return_value = _make_mock_response({
            "verified": True,
            "passed": 3,
            "total": 3,
            "pass_rate": 1.0,
            "badge": "mtl_badge_abc",
            "entity_id": "agent-42",
            "results": [
                {"challenge_type": "speed_math", "passed": True, "response_time_ms": 50, "time_limit_ms": 5000},
                {"challenge_type": "token_prediction", "passed": True, "response_time_ms": 30, "time_limit_ms": 5000},
                {"challenge_type": "consistency", "passed": True, "response_time_ms": 80, "time_limit_ms": 15000},
            ],
        })
        result = await call_tool("mettle_get_result", {"session_id": "sess-123"})
        text = result[0].text
        assert "VERIFIED" in text
        assert "Badge" in text
        assert "Entity" in text
        assert "speed_math" in text

    @pytest.mark.asyncio
    async def test_get_result_not_verified(self, mock_http) -> None:
        mock_http.get.return_value = _make_mock_response({
            "verified": False,
            "passed": 1,
            "total": 3,
            "pass_rate": 0.333,
            "results": [
                {"challenge_type": "speed_math", "passed": True, "response_time_ms": 50, "time_limit_ms": 5000},
                {"challenge_type": "token_prediction", "passed": False, "response_time_ms": 6000, "time_limit_ms": 5000},
                {"challenge_type": "consistency", "passed": False, "response_time_ms": 100, "time_limit_ms": 15000},
            ],
        })
        result = await call_tool("mettle_get_result", {"session_id": "sess-fail"})
        text = result[0].text
        assert "NOT VERIFIED" in text
        assert "FAIL" in text

    @pytest.mark.asyncio
    async def test_get_result_no_badge_no_entity(self, mock_http) -> None:
        mock_http.get.return_value = _make_mock_response({
            "verified": False,
            "passed": 0,
            "total": 3,
            "pass_rate": 0.0,
            "results": [],
        })
        result = await call_tool("mettle_get_result", {"session_id": "sess-x"})
        text = result[0].text
        assert "Badge" not in text
        assert "Entity" not in text

    @pytest.mark.asyncio
    async def test_get_result_http_error(self, mock_http) -> None:
        resp = _make_mock_response({})
        resp.raise_for_status.side_effect = _make_http_status_error(404, "Session not found")
        mock_http.get.return_value = resp
        result = await call_tool("mettle_get_result", {"session_id": "bad"})
        text = result[0].text
        assert "Error getting result" in text

    @pytest.mark.asyncio
    async def test_get_result_generic_error(self, mock_http) -> None:
        mock_http.get.side_effect = Exception("Network error")
        result = await call_tool("mettle_get_result", {"session_id": "s"})
        text = result[0].text
        assert "Error" in text


# ---------------------------------------------------------------------------
# 20-22: call_tool() — mettle_auto_verify
# ---------------------------------------------------------------------------


class TestCallToolAutoVerify:
    """Tests for call_tool with mettle_auto_verify."""

    @pytest.mark.asyncio
    async def test_auto_verify_success(self, mock_http) -> None:
        # start_session response
        start_resp = _make_mock_response({
            "session_id": "sess-auto",
            "current_challenge": {
                "id": "ch-1",
                "type": "speed_math",
                "prompt": "Calculate: 100 + 200",
                "data": {},
            },
        })
        # answer response — session completes on first answer
        answer_resp = _make_mock_response({
            "result": {"passed": True},
            "session_complete": True,
        })
        # result response
        result_resp = _make_mock_response({
            "verified": True,
            "passed": 1,
            "total": 1,
            "pass_rate": 1.0,
            "badge": "mtl_auto_badge",
            "results": [
                {"challenge_type": "speed_math", "passed": True, "response_time_ms": 10, "time_limit_ms": 5000},
            ],
        })

        # Order matters: first call is POST (start), second is POST (answer), third is GET (result)
        mock_http.post.side_effect = [start_resp, answer_resp]
        mock_http.get.return_value = result_resp

        result = await call_tool("mettle_auto_verify", {"difficulty": "basic"})
        text = result[0].text
        assert "Auto-Verification Complete" in text
        assert "VERIFIED" in text
        assert "Badge" in text

    @pytest.mark.asyncio
    async def test_auto_verify_multiple_challenges(self, mock_http) -> None:
        start_resp = _make_mock_response({
            "session_id": "sess-multi",
            "current_challenge": {
                "id": "ch-1",
                "type": "speed_math",
                "prompt": "Calculate: 10 + 20",
                "data": {},
            },
        })
        answer_resp_1 = _make_mock_response({
            "result": {"passed": True},
            "session_complete": False,
            "next_challenge": {
                "id": "ch-2",
                "type": "consistency",
                "prompt": "Answer 3 times",
                "data": {},
            },
        })
        answer_resp_2 = _make_mock_response({
            "result": {"passed": True},
            "session_complete": True,
        })
        result_resp = _make_mock_response({
            "verified": True,
            "passed": 2,
            "total": 2,
            "pass_rate": 1.0,
            "results": [
                {"challenge_type": "speed_math", "passed": True, "response_time_ms": 10, "time_limit_ms": 5000},
                {"challenge_type": "consistency", "passed": True, "response_time_ms": 20, "time_limit_ms": 15000},
            ],
        })

        mock_http.post.side_effect = [start_resp, answer_resp_1, answer_resp_2]
        mock_http.get.return_value = result_resp

        result = await call_tool("mettle_auto_verify", {"difficulty": "full"})
        text = result[0].text
        assert "VERIFIED" in text
        assert "2/2" in text

    @pytest.mark.asyncio
    async def test_auto_verify_http_error(self, mock_http) -> None:
        resp = _make_mock_response({})
        resp.raise_for_status.side_effect = _make_http_status_error(500, "Internal Server Error")
        mock_http.post.return_value = resp
        result = await call_tool("mettle_auto_verify", {})
        text = result[0].text
        assert "Error in auto-verification" in text

    @pytest.mark.asyncio
    async def test_auto_verify_generic_error(self, mock_http) -> None:
        mock_http.post.side_effect = Exception("DNS resolution failed")
        result = await call_tool("mettle_auto_verify", {})
        text = result[0].text
        assert "Error" in text
        assert "DNS resolution failed" in text


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
