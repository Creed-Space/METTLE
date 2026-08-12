"""Tests for mettle.mcp_server — the packaged MCP server.

Covers:
- A real stdio round-trip: spawn the server as a subprocess, run the MCP
  handshake, and assert tools/list returns the expected tool set.
- The per-session auth path: /session/* calls must carry X-Session-Token,
  which the live API enforces with a 401. This is the path that regressed
  when the v2 server was written, so it is asserted directly rather than
  being left to the tools/list smoke check.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# The MCP server is an optional extra (`pip install 'mettle-verifier[mcp]'`).
# Importing mettle.mcp_server pulls in `mcp` at collection time, so guard the
# whole module rather than letting a missing extra error out the entire run.
# CI installs requirements-mcp.txt, so these do not silently skip there.
pytest.importorskip("mcp")

from mettle import mcp_server  # noqa: E402

EXPECTED_TOOLS = {
    "mettle_start_session",
    "mettle_answer_challenge",
    "mettle_get_result",
    "mettle_list_suites",
    "mettle_start_v2_session",
    "mettle_verify_suite",
    "mettle_get_v2_result",
}


# === stdio smoke test ===


@pytest.mark.asyncio
async def test_stdio_server_lists_tools():
    """Spawn the server over stdio and assert the advertised tool names."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mettle.mcp_server"],
        env={**os.environ, "METTLE_API_URL": "https://mettle.sh/api"},
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()

    assert {t.name for t in result.tools} == EXPECTED_TOOLS


# === per-session auth path ===


def _mock_response(payload: object) -> MagicMock:
    """Build a stub httpx response. Payload is object, not dict: /mettle/suites
    returns a JSON array."""
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def _headers(mock: AsyncMock) -> dict:
    """Headers of the most recent await, asserting one actually happened."""
    call = mock.await_args
    assert call is not None, "expected an HTTP call to have been awaited"
    return call.kwargs["headers"]


@pytest.mark.asyncio
async def test_session_token_sent_as_header_on_post():
    """api_call must present the session token the MVP endpoints require."""
    post = AsyncMock(return_value=_mock_response({"ok": True}))
    with patch.object(mcp_server.http_client, "post", post):
        await mcp_server.api_call(
            "/session/answer", "POST", {"answer": "42"}, session_token="tok_abc"
        )

    assert _headers(post)["X-Session-Token"] == "tok_abc"


@pytest.mark.asyncio
async def test_session_token_sent_as_header_on_get():
    get = AsyncMock(return_value=_mock_response({"ok": True}))
    with patch.object(mcp_server.http_client, "get", get):
        await mcp_server.api_call("/session/x/result", session_token="tok_xyz")

    assert _headers(get)["X-Session-Token"] == "tok_xyz"


@pytest.mark.asyncio
async def test_no_session_token_means_no_header():
    """The v2 endpoints must not be given a stray session header."""
    get = AsyncMock(return_value=_mock_response({"ok": True}))
    with patch.object(mcp_server.http_client, "get", get):
        await mcp_server.api_call("/mettle/suites")

    assert "X-Session-Token" not in _headers(get)


@pytest.mark.asyncio
async def test_answer_challenge_threads_token_through():
    """The tool handler must forward the caller's token, not drop it."""
    post = AsyncMock(
        return_value=_mock_response(
            {
                "result": {
                    "passed": True,
                    "response_time_ms": 10,
                    "time_limit_ms": 2500,
                },
                "session_complete": True,
            }
        )
    )
    with patch.object(mcp_server.http_client, "post", post):
        await mcp_server.call_tool(
            "mettle_answer_challenge",
            {
                "session_id": "ses_1",
                "session_token": "tok_from_start",
                "challenge_id": "mtl_1",
                "answer": "44",
            },
        )

    assert _headers(post)["X-Session-Token"] == "tok_from_start"


@pytest.mark.asyncio
async def test_v2_endpoint_requires_api_key():
    with patch.object(mcp_server, "API_KEY", None):
        with pytest.raises(RuntimeError, match="METTLE_API_KEY"):
            await mcp_server.api_call("/mettle/suites", auth=True)


@pytest.mark.parametrize("bad", ["../etc", "a/b", "x?y", "", "z" * 65])
def test_safe_id_rejects_path_injection(bad):
    with pytest.raises(ValueError):
        mcp_server._safe_id(bad, "session_id")


# === per-tool happy paths and error handling ===
#
# The module this file replaced covered the old root server's tools and error
# branches; that breadth is retained here against the packaged module so the
# repo coverage gate still holds.


def _http_error(
    status_code: int = 400, text: str = "Bad Request"
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://mettle.sh/api/x")
    response = httpx.Response(status_code, request=request, text=text)
    return httpx.HTTPStatusError(text, request=request, response=response)


def _text(result) -> str:
    return "".join(chunk.text for chunk in result)


START_PAYLOAD = {
    "session_id": "ses_1",
    "session_token": "tok_1",
    "difficulty": "basic",
    "total_challenges": 3,
    "current_challenge": {
        "id": "mtl_1",
        "type": "speed_math",
        "prompt": "Calculate: 2 + 2",
        "data": {"a": 2, "b": 2, "op": "+"},
        "time_limit_ms": 2500,
    },
}


@pytest.mark.asyncio
async def test_start_session_reports_ids_and_first_challenge():
    post = AsyncMock(return_value=_mock_response(START_PAYLOAD))
    with patch.object(mcp_server.http_client, "post", post):
        text = _text(await mcp_server.call_tool("mettle_start_session", {}))

    assert "ses_1" in text
    assert "tok_1" in text
    assert "Calculate: 2 + 2" in text


@pytest.mark.asyncio
async def test_answer_challenge_reports_next_challenge():
    payload = {
        "result": {"passed": True, "response_time_ms": 12, "time_limit_ms": 2500},
        "session_complete": False,
        "challenges_remaining": 2,
        "next_challenge": {
            "id": "mtl_2",
            "type": "parallel_recall",
            "prompt": "Next one",
            "time_limit_ms": 3000,
        },
    }
    post = AsyncMock(return_value=_mock_response(payload))
    with patch.object(mcp_server.http_client, "post", post):
        text = _text(
            await mcp_server.call_tool(
                "mettle_answer_challenge",
                {
                    "session_id": "ses_1",
                    "session_token": "tok_1",
                    "challenge_id": "mtl_1",
                    "answer": "4",
                },
            )
        )

    assert "PASSED" in text
    assert "mtl_2" in text
    assert "Challenges remaining: 2" in text


@pytest.mark.asyncio
async def test_get_result_renders_badge_and_per_challenge_rows():
    payload = {
        "verified": True,
        "passed": 3,
        "total": 3,
        "pass_rate": 1.0,
        "badge": "mettle:bronze",
        "entity_id": "agent-7",
        "results": [
            {
                "challenge_type": "speed_math",
                "passed": True,
                "response_time_ms": 10,
                "time_limit_ms": 2500,
            }
        ],
    }
    get = AsyncMock(return_value=_mock_response(payload))
    with patch.object(mcp_server.http_client, "get", get):
        text = _text(
            await mcp_server.call_tool(
                "mettle_get_result", {"session_id": "ses_1", "session_token": "tok_1"}
            )
        )

    assert "VERIFIED" in text
    assert "mettle:bronze" in text
    assert "agent-7" in text
    assert "speed_math" in text


@pytest.mark.asyncio
async def test_list_suites_renders_flags():
    payload = [
        {
            "suite_number": 1,
            "name": "speed",
            "display_name": "Inhuman Speed",
            "description": "Sub-human latency",
            "is_multi_round": True,
            "available": False,
        }
    ]
    get = AsyncMock(return_value=_mock_response(payload))
    with patch.object(mcp_server, "API_KEY", "mtl_test"):
        with patch.object(mcp_server.http_client, "get", get):
            text = _text(await mcp_server.call_tool("mettle_list_suites", {}))

    assert "Inhuman Speed" in text
    assert "multi-round" in text
    assert "unavailable" in text
    assert _headers(get)["Authorization"] == "Bearer mtl_test"


@pytest.mark.asyncio
async def test_start_v2_session_passes_optional_fields():
    payload = {
        "session_id": "ses_v2",
        "suites": ["speed"],
        "time_budget_ms": 60000,
        "expires_at": "2026-07-19T02:00:00Z",
        "challenges": {"speed": {"items": [1, 2]}},
    }
    post = AsyncMock(return_value=_mock_response(payload))
    with patch.object(mcp_server, "API_KEY", "mtl_test"):
        with patch.object(mcp_server.http_client, "post", post):
            text = _text(
                await mcp_server.call_tool(
                    "mettle_start_v2_session",
                    {
                        "suites": ["speed"],
                        "entity_id": "agent-7",
                        "vcp_token": "CSM1.abc",
                    },
                )
            )

    call = post.await_args
    assert call is not None
    body = call.kwargs["json"]
    assert body["entity_id"] == "agent-7"
    assert body["vcp_token"] == "CSM1.abc"
    assert "ses_v2" in text


@pytest.mark.asyncio
async def test_verify_suite_reports_score_and_details():
    payload = {"suite": "speed", "passed": True, "score": 0.91, "details": {"n": 4}}
    post = AsyncMock(return_value=_mock_response(payload))
    with patch.object(mcp_server, "API_KEY", "mtl_test"):
        with patch.object(mcp_server.http_client, "post", post):
            text = _text(
                await mcp_server.call_tool(
                    "mettle_verify_suite",
                    {"session_id": "ses_v2", "suite": "speed", "answers": {"a": 1}},
                )
            )

    assert "PASSED" in text
    assert "0.91" in text
    assert '"n": 4' in text


@pytest.mark.asyncio
async def test_get_v2_result_renders_attestations():
    payload = {
        "status": "complete",
        "overall_passed": True,
        "tier": "gold",
        "suites_completed": ["speed", "parallel"],
        "vcp_attestation": {"tier": "gold", "sig": "ed25519:abc"},
        "governance_attestation": {"framework": "VCP"},
    }
    get = AsyncMock(return_value=_mock_response(payload))
    with patch.object(mcp_server, "API_KEY", "mtl_test"):
        with patch.object(mcp_server.http_client, "get", get):
            text = _text(
                await mcp_server.call_tool(
                    "mettle_get_v2_result", {"session_id": "ses_v2"}
                )
            )

    assert "PASSED" in text
    assert "gold" in text
    assert "ed25519:abc" in text
    assert "framework=VCP" in text
    params_call = get.await_args
    assert params_call is not None
    assert params_call.kwargs["params"] == {"include_vcp": "true"}


@pytest.mark.asyncio
async def test_unknown_tool_is_reported():
    text = _text(await mcp_server.call_tool("mettle_nope", {}))
    assert "Unknown tool" in text


@pytest.mark.asyncio
async def test_auto_solver_is_not_exposed():
    """An MCP client must solve its own challenges before any credential can issue."""
    assert "mettle_auto_verify" not in {
        tool.name for tool in await mcp_server.list_tools()
    }
    text = _text(await mcp_server.call_tool("mettle_auto_verify", {}))
    assert "Unknown tool" in text


@pytest.mark.asyncio
async def test_mcp2_call_adapter_returns_protocol_content():
    """The low-level MCP 2 handler preserves the transport-independent response."""
    from mcp.types import CallToolRequestParams

    result = await mcp_server._handle_call_tool(
        MagicMock(), CallToolRequestParams(name="mettle_nope", arguments={})
    )
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert "Unknown tool" in result.content[0].text


# Every tool surfaces both an HTTP error and an unexpected error as text rather
# than raising through the MCP transport.
TOOL_CASES = [
    ("mettle_start_session", {}, "post"),
    (
        "mettle_answer_challenge",
        {
            "session_id": "s",
            "session_token": "t",
            "challenge_id": "c",
            "answer": "a",
        },
        "post",
    ),
    ("mettle_get_result", {"session_id": "s", "session_token": "t"}, "get"),
    ("mettle_list_suites", {}, "get"),
    ("mettle_start_v2_session", {}, "post"),
    (
        "mettle_verify_suite",
        {"session_id": "s", "suite": "speed", "answers": {}},
        "post",
    ),
    ("mettle_get_v2_result", {"session_id": "s"}, "get"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,args,verb", TOOL_CASES)
async def test_tools_report_http_errors_as_text(tool, args, verb):
    mock = AsyncMock(side_effect=_http_error(400, "boom"))
    with patch.object(mcp_server, "API_KEY", "mtl_test"):
        with patch.object(mcp_server.http_client, verb, mock):
            text = _text(await mcp_server.call_tool(tool, args))

    assert "Error" in text
    assert "boom" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,args,verb", TOOL_CASES)
async def test_tools_report_unexpected_errors_as_text(tool, args, verb):
    mock = AsyncMock(side_effect=RuntimeError("kaboom"))
    with patch.object(mcp_server, "API_KEY", "mtl_test"):
        with patch.object(mcp_server.http_client, verb, mock):
            text = _text(await mcp_server.call_tool(tool, args))

    assert "Error" in text
    assert "kaboom" in text


@pytest.mark.asyncio
async def test_get_result_rejects_injected_session_id():
    text = _text(
        await mcp_server.call_tool(
            "mettle_get_result", {"session_id": "../admin", "session_token": "t"}
        )
    )
    assert "invalid session_id" in text
