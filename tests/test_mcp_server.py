"""Adversarial contract tests for the bounded MCP adapter."""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

try:
    import mcp  # noqa: F401
except ImportError:

    class _FakeTextContent:
        def __init__(self, **values: Any):
            self.__dict__.update(values)

    class _FakeTool:
        def __init__(self, **values: Any):
            self.__dict__.update(values)

    class _FakeCallToolResult:
        def __init__(self, **values: Any):
            self.__dict__.update(values)
            self.is_error = values.get("isError", False)

    class _FakeServer:
        def __init__(self, name: str):
            self.name = name

        def list_tools(self):
            return lambda function: function

        def call_tool(self):
            return lambda function: function

        def create_initialization_options(self):
            return object()

    sys.modules["mcp"] = MagicMock()
    sys.modules["mcp.server"] = MagicMock()
    sys.modules["mcp.server.stdio"] = MagicMock()
    sys.modules["mcp.types"] = MagicMock()
    cast(Any, sys.modules["mcp.server"]).Server = _FakeServer
    cast(Any, sys.modules["mcp.types"]).CallToolResult = _FakeCallToolResult
    cast(Any, sys.modules["mcp.types"]).TextContent = _FakeTextContent
    cast(Any, sys.modules["mcp.types"]).Tool = _FakeTool

import mcp_server  # noqa: E402


SESSION_ID = "ses_" + "a" * 24
CHALLENGE_ID = "mtl_" + "b" * 24
SESSION_TOKEN = "T" * 43


def _text(result: Any) -> str:
    assert len(result.content) == 1
    return result.content[0].text


def _is_error(result: Any) -> bool:
    return bool(getattr(result, "isError", getattr(result, "is_error", False)))


def _answer_payload(answer: str = "42") -> dict[str, str]:
    return {
        "session_id": SESSION_ID,
        "session_token": SESSION_TOKEN,
        "challenge_id": CHALLENGE_ID,
        "answer": answer,
    }


def _start_response() -> dict[str, Any]:
    return {
        "session_id": SESSION_ID,
        "session_token": SESSION_TOKEN,
        "difficulty": "basic",
        "total_challenges": 3,
        "current_challenge": {
            "id": CHALLENGE_ID,
            "type": "speed_math",
            "prompt": "Calculate: 1 + 1",
            "time_limit_ms": 5000,
        },
    }


def _answer_response(*, complete: bool, passed: bool = True) -> dict[str, Any]:
    response: dict[str, Any] = {
        "result": {
            "passed": passed,
            "response_time_ms": 50,
            "time_limit_ms": 5000,
        },
        "session_complete": complete,
    }
    if not complete:
        response.update(
            {
                "challenges_remaining": 2,
                "next_challenge": {
                    "id": "mtl_" + "c" * 24,
                    "type": "token_prediction",
                    "prompt": "Complete: hello",
                    "time_limit_ms": 5000,
                },
            }
        )
    return response


async def _api_with_transport(
    handler: Any,
    endpoint: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with patch.object(mcp_server, "http_client", client):
            return await mcp_server.api_call(endpoint, method, payload, token)


class TestApiCall:
    @pytest.mark.asyncio
    async def test_get_uses_exact_url_and_owner_header(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "GET"
            assert str(request.url) == "https://mettle.sh/api/health"
            assert request.headers["X-Session-Token"] == SESSION_TOKEN
            assert request.content == b""
            return httpx.Response(200, json={"status": "ok"})

        assert await _api_with_transport(handler, "/health", token=SESSION_TOKEN) == {
            "status": "ok"
        }

    @pytest.mark.asyncio
    async def test_post_uses_exact_json_without_token_leakage(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "POST"
            assert str(request.url) == "https://mettle.sh/api/session/start"
            assert "X-Session-Token" not in request.headers
            assert json.loads(request.content) == {"difficulty": "basic"}
            return httpx.Response(200, json={"status": "ok"})

        result = await _api_with_transport(
            handler, "/session/start", "POST", {"difficulty": "basic"}
        )
        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_exact_response_byte_limit_is_accepted(self) -> None:
        overhead = len(b'{"value":""}')
        body = (
            b'{"value":"'
            + b"x" * (mcp_server.MAX_UPSTREAM_RESPONSE_BYTES - overhead)
            + b'"}'
        )
        assert len(body) == mcp_server.MAX_UPSTREAM_RESPONSE_BYTES

        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body)

        result = await _api_with_transport(handler, "/health")
        assert len(result["value"]) == len(body) - overhead

    @pytest.mark.asyncio
    async def test_response_over_byte_limit_is_rejected(self) -> None:
        body = b"X" * (mcp_server.MAX_UPSTREAM_RESPONSE_BYTES + 1)

        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body)

        with pytest.raises(mcp_server.UpstreamResponseError, match="exceeded"):
            await _api_with_transport(handler, "/health")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body", [b"[]", b"not-json", b'{"x":1,"x":2}', b'{"x":NaN}']
    )
    async def test_invalid_or_non_object_json_is_rejected(self, body: bytes) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body)

        with pytest.raises(mcp_server.UpstreamResponseError):
            await _api_with_transport(handler, "/health")


class TestToolSchemas:
    @pytest.mark.asyncio
    async def test_complete_strict_schemas_are_advertised_once(self) -> None:
        tools = await mcp_server.list_tools()
        by_name = {tool.name: tool.inputSchema for tool in tools}
        assert set(by_name) == set(mcp_server.TOOL_INPUT_SCHEMAS)
        assert len(tools) == len(by_name) == 3
        for name, schema in by_name.items():
            assert schema == mcp_server.TOOL_INPUT_SCHEMAS[name]
            assert schema["additionalProperties"] is False
        answer = by_name["mettle_answer_challenge"]
        assert set(answer["required"]) == set(answer["properties"])
        assert answer["properties"]["answer"]["maxLength"] == 1024
        assert answer["properties"]["session_id"]["pattern"].startswith("^ses_")


class TestRuntimeValidation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("name", "arguments"),
        [
            ("mettle_nonexistent", {}),
            ("mettle_start_session", []),
            ("mettle_start_session", {"difficulty": "wrong"}),
            ("mettle_start_session", {"unexpected": True}),
            ("mettle_start_session", {"entity_id": "x" * 129}),
            ("mettle_answer_challenge", {}),
            (
                "mettle_answer_challenge",
                {**_answer_payload(), "session_id": "ses_../" + "a" * 21},
            ),
            (
                "mettle_answer_challenge",
                {**_answer_payload(), "session_token": 123},
            ),
            (
                "mettle_answer_challenge",
                {**_answer_payload(), "answer": "x" * 1025},
            ),
            (
                "mettle_get_result",
                {
                    "session_id": "ses_../" + "a" * 21,
                    "session_token": SESSION_TOKEN,
                },
            ),
        ],
    )
    async def test_invalid_calls_are_errors_without_upstream_calls(
        self, name: str, arguments: Any
    ) -> None:
        with patch.object(mcp_server, "api_call", new_callable=AsyncMock) as api:
            result = await mcp_server.call_tool(name, arguments)
        assert _is_error(result) is True
        assert _text(result).startswith("Invalid METTLE tool call:")
        api.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exact_entity_and_answer_maxima_are_forwarded(self) -> None:
        entity_id = "a" + "b" * 127
        with patch.object(mcp_server, "api_call", new_callable=AsyncMock) as api:
            api.side_effect = [_start_response(), _answer_response(complete=True)]
            start = await mcp_server.call_tool(
                "mettle_start_session", {"entity_id": entity_id}
            )
            answer = await mcp_server.call_tool(
                "mettle_answer_challenge", _answer_payload("x" * 1024)
            )
        assert _is_error(start) is False
        assert _is_error(answer) is False
        assert api.await_count == 2


class TestToolDispatch:
    @pytest.mark.asyncio
    async def test_start_session_exact_request_and_rendering(self) -> None:
        with patch.object(mcp_server, "api_call", new_callable=AsyncMock) as api:
            api.return_value = _start_response()
            result = await mcp_server.call_tool(
                "mettle_start_session",
                {"difficulty": "basic", "entity_id": "agent-1"},
            )
        api.assert_awaited_once_with(
            "/session/start",
            "POST",
            {"difficulty": "basic", "entity_id": "agent-1"},
        )
        assert _is_error(result) is False
        assert SESSION_ID in _text(result)
        assert "speed_math" in _text(result)

    @pytest.mark.asyncio
    async def test_answer_exact_request_keeps_token_out_of_body(self) -> None:
        with patch.object(mcp_server, "api_call", new_callable=AsyncMock) as api:
            api.return_value = _answer_response(complete=False)
            result = await mcp_server.call_tool(
                "mettle_answer_challenge", _answer_payload()
            )
        api.assert_awaited_once_with(
            "/session/answer",
            "POST",
            {
                "session_id": SESSION_ID,
                "challenge_id": CHALLENGE_ID,
                "answer": "42",
            },
            session_token=SESSION_TOKEN,
        )
        assert _is_error(result) is False
        assert "PASSED" in _text(result)
        assert "token_prediction" in _text(result)

    @pytest.mark.asyncio
    async def test_complete_failed_answer_has_no_next_challenge(self) -> None:
        with patch.object(mcp_server, "api_call", new_callable=AsyncMock) as api:
            api.return_value = _answer_response(complete=True, passed=False)
            result = await mcp_server.call_tool(
                "mettle_answer_challenge", _answer_payload()
            )
        assert "FAILED" in _text(result)
        assert "Session complete" in _text(result)
        assert "Next Challenge" not in _text(result)

    @pytest.mark.asyncio
    async def test_get_result_exact_safe_path_and_rendering(self) -> None:
        response = {
            "verified": True,
            "tier": "bronze",
            "passed": 1,
            "total": 1,
            "pass_rate": 1.0,
            "entity_id": "agent-1",
            "badge": "signed-badge",
            "results": [
                {
                    "challenge_type": "speed_math",
                    "passed": True,
                    "response_time_ms": 50,
                    "time_limit_ms": 5000,
                }
            ],
        }
        with patch.object(mcp_server, "api_call", new_callable=AsyncMock) as api:
            api.return_value = response
            result = await mcp_server.call_tool(
                "mettle_get_result",
                {"session_id": SESSION_ID, "session_token": SESSION_TOKEN},
            )
        api.assert_awaited_once_with(
            f"/session/{SESSION_ID}/result", session_token=SESSION_TOKEN
        )
        text = _text(result)
        assert _is_error(result) is False
        assert "VERIFIED" in text
        assert "signed-badge" in text
        assert "agent-1" in text

    @pytest.mark.asyncio
    async def test_unverified_result_omits_optional_identity_and_badge(self) -> None:
        response = {
            "verified": False,
            "passed": 0,
            "total": 1,
            "pass_rate": 0.0,
            "results": [
                {
                    "challenge_type": "speed_math",
                    "passed": False,
                    "response_time_ms": 5001,
                    "time_limit_ms": 5000,
                }
            ],
        }
        with patch.object(mcp_server, "api_call", new_callable=AsyncMock) as api:
            api.return_value = response
            result = await mcp_server.call_tool(
                "mettle_get_result",
                {"session_id": SESSION_ID, "session_token": SESSION_TOKEN},
            )
        text = _text(result)
        assert _is_error(result) is False
        assert "NOT VERIFIED" in text
        assert "FAIL" in text
        assert "Entity:" not in text
        assert "Signed credential" not in text


class TestToolErrors:
    @pytest.mark.asyncio
    async def test_http_error_is_explicit_and_redacts_remote_body(self) -> None:
        request = httpx.Request("POST", "https://mettle.invalid")
        response = httpx.Response(
            500,
            request=request,
            text="secret upstream diagnostic" * 50_000,
        )
        error = httpx.HTTPStatusError(
            "secret exception", request=request, response=response
        )
        with patch.object(
            mcp_server, "api_call", new_callable=AsyncMock, side_effect=error
        ):
            result = await mcp_server.call_tool("mettle_start_session", {})
        assert _is_error(result) is True
        assert _text(result) == "METTLE API rejected the request."
        assert "secret" not in _text(result)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            httpx.ReadTimeout("secret timeout detail"),
            mcp_server.UpstreamResponseError("secret response detail"),
            RuntimeError("secret internal detail"),
        ],
    )
    async def test_transport_and_internal_errors_are_bounded_and_redacted(
        self, error: Exception
    ) -> None:
        with patch.object(
            mcp_server, "api_call", new_callable=AsyncMock, side_effect=error
        ):
            result = await mcp_server.call_tool("mettle_start_session", {})
        assert _is_error(result) is True
        assert "secret" not in _text(result)
        assert len(_text(result)) < 100

    @pytest.mark.asyncio
    async def test_invalid_upstream_shape_is_explicit_error(self) -> None:
        with patch.object(mcp_server, "api_call", new_callable=AsyncMock) as api:
            api.return_value = {"current_challenge": []}
            result = await mcp_server.call_tool("mettle_start_session", {})
        assert _is_error(result) is True
        assert _text(result) == "METTLE API returned an invalid response."

    @pytest.mark.asyncio
    async def test_unissued_upstream_difficulty_is_an_explicit_error(self) -> None:
        response = _start_response()
        response["difficulty"] = "other"
        with patch.object(mcp_server, "api_call", new_callable=AsyncMock) as api:
            api.return_value = response
            result = await mcp_server.call_tool("mettle_start_session", {})
        assert _is_error(result) is True
        assert _text(result) == "METTLE API returned an invalid response."

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response",
        [
            {
                "verified": "false",
                "passed": 0,
                "total": 0,
                "pass_rate": 0.0,
                "results": [],
            },
            {
                "verified": False,
                "passed": 1,
                "total": 3,
                "pass_rate": 1.0,
                "results": [],
            },
            {
                "verified": False,
                "tier": "platinum",
                "passed": 0,
                "total": 0,
                "pass_rate": 0.0,
                "results": [],
            },
            {
                "verified": False,
                "tier": "none",
                "passed": 0,
                "total": 1,
                "pass_rate": 0.0,
                "results": [],
            },
        ],
    )
    async def test_semantically_invalid_result_is_not_rendered_as_success(
        self, response: dict[str, Any]
    ) -> None:
        with patch.object(mcp_server, "api_call", new_callable=AsyncMock) as api:
            api.return_value = response
            result = await mcp_server.call_tool(
                "mettle_get_result",
                {"session_id": SESSION_ID, "session_token": SESSION_TOKEN},
            )
        assert _is_error(result) is True
        assert _text(result) == "METTLE API returned an invalid response."


@pytest.mark.asyncio
async def test_main_closes_http_client_when_server_run_fails() -> None:
    @asynccontextmanager
    async def fake_stdio():
        yield object(), object()

    client = MagicMock()
    client.aclose = AsyncMock()
    fake_server = MagicMock()
    fake_server.create_initialization_options.return_value = object()
    fake_server.run = AsyncMock(side_effect=RuntimeError("server stopped"))
    with (
        patch.object(mcp_server, "stdio_server", fake_stdio),
        patch.object(mcp_server, "http_client", client),
        patch.object(mcp_server, "server", fake_server),
        pytest.raises(RuntimeError, match="server stopped"),
    ):
        await mcp_server.main()
    client.aclose.assert_awaited_once_with()
