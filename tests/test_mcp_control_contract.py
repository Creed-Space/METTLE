"""Agent-first acceptance and security checks for MCP control-v1."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import jsonschema  # type: ignore[import-untyped]
import pytest
from fastapi import FastAPI
from mcp.types import CallToolRequestParams

from mettle import mcp_server
from mettle.auth import require_authenticated_user
from mettle.mcp_contract import CONTROL_OUTPUT_SCHEMA, SCHEMA_VERSION
from mettle.router import get_session_manager, router
from mettle.session_manager import SessionManager
from tests.test_rewind_integration import FakeRedis, _make_mock_user


@pytest.fixture(autouse=True)
def clear_session_token_vault():
    mcp_server._session_tokens.clear()
    yield
    mcp_server._session_tokens.clear()


def _response(payload: object, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.content = b"" if status_code == 204 else b"{}"
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def _assert_valid(response) -> dict:
    envelope = response.structured_content
    jsonschema.Draft202012Validator(CONTROL_OUTPUT_SCHEMA).validate(envelope)
    assert envelope["schema_version"] == SCHEMA_VERSION
    return envelope


@pytest.mark.asyncio
async def test_every_tool_publishes_output_schema_and_effect_annotations():
    tools = {tool.name: tool for tool in await mcp_server.list_tools()}

    assert len(tools) == 11
    assert "mettle_auto_verify" not in tools
    for tool in tools.values():
        assert tool.output_schema == CONTROL_OUTPUT_SCHEMA
        assert tool.annotations is not None
        assert tool.annotations.open_world_hint is True

    for name in (
        "mettle_list_suites",
        "mettle_get_round_feedback",
    ):
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.read_only_hint is True
        assert annotations.idempotent_hint is True
        assert annotations.destructive_hint is False

    for name in ("mettle_get_result", "mettle_get_v2_result", "mettle_get_session"):
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.read_only_hint is False
        assert annotations.destructive_hint is False
        assert annotations.idempotent_hint is True

    for name in (
        "mettle_answer_challenge",
        "mettle_verify_suite",
        "mettle_cancel_session",
        "mettle_submit_round",
    ):
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.read_only_hint is False
        assert annotations.destructive_hint is True
        assert annotations.idempotent_hint is False


@pytest.mark.asyncio
async def test_quick_start_structured_content_never_exposes_hidden_bearer():
    payload = {
        "session_id": "ses_1",
        "session_token": "secret-token-value",
        "difficulty": "basic",
        "total_challenges": 3,
        "current_challenge": {
            "id": "mtl_1",
            "type": "speed_math",
            "prompt": "Calculate 2 + 2",
            "time_limit_ms": 2500,
        },
    }
    with patch.object(mcp_server, "api_call", AsyncMock(return_value=payload)):
        result = await mcp_server.call_tool("mettle_start_session", {})

    envelope = _assert_valid(result)
    serialized = json.dumps(envelope)
    assert "secret-token-value" not in serialized
    assert "session_token" not in serialized
    assert mcp_server._get_session_token("ses_1") == "secret-token-value"
    assert envelope["snapshot"]["current_challenge"]["id"] == "mtl_1"
    assert envelope["actions"][0]["operation"] == "mettle_answer_challenge"


@pytest.mark.asyncio
async def test_quick_result_is_repeatable_with_one_hidden_capability():
    payload = {
        "verified": True,
        "passed": 3,
        "total": 3,
        "pass_rate": 1.0,
        "results": [],
    }
    call = AsyncMock(return_value=payload)
    mcp_server._remember_session_token("ses_1", "tok_1")
    with patch.object(mcp_server, "api_call", call):
        first = await mcp_server.call_tool("mettle_get_result", {"session_id": "ses_1"})
        second = await mcp_server.call_tool(
            "mettle_get_result", {"session_id": "ses_1"}
        )

    assert _assert_valid(first)["data"] == _assert_valid(second)["data"]
    assert call.await_count == 2
    assert mcp_server._get_session_token("ses_1") == "tok_1"


@pytest.mark.asyncio
async def test_protocol_error_is_bounded_structured_and_non_reflective():
    request = httpx.Request("POST", "https://mettle.sh/api/mettle/sessions")
    response = httpx.Response(
        400,
        request=request,
        text="private upstream stack and supplied participant content",
    )
    error = httpx.HTTPStatusError(
        "private exception", request=request, response=response
    )
    with patch.object(mcp_server, "api_call", AsyncMock(side_effect=error)):
        result = await mcp_server._handle_call_tool(
            MagicMock(),
            CallToolRequestParams(name="mettle_start_v2_session", arguments={}),
        )

    assert result.is_error is True
    serialized = json.dumps(result.structured_content)
    assert "private" not in serialized
    assert "participant content" not in serialized
    assert result.structured_content["error"] == {
        "code": "invalid_request",
        "message": "The request was rejected by the session authority.",
        "retry": "refresh_then_retry",
        "http_status": 400,
    }
    assert len(serialized) < 1000


@pytest.mark.parametrize(
    ("operation", "expected_retry"),
    [
        ("mettle_get_session", "safe_same_operation"),
        ("mettle_submit_round", "refresh_then_retry"),
        ("mettle_start_v2_session", "operator_action_required"),
    ],
)
def test_timeout_retry_guidance_never_blindly_repeats_an_ambiguous_mutation(
    operation: str, expected_retry: str
):
    result = mcp_server.failure(operation, httpx.ReadTimeout("timed out"))

    assert _assert_valid(result)["error"]["retry"] == expected_retry


@pytest.mark.parametrize(
    ("operation", "expected_retry"),
    [
        ("mettle_get_session", "retry_after"),
        ("mettle_verify_suite", "refresh_then_retry"),
        ("mettle_start_session", "operator_action_required"),
    ],
)
def test_server_error_retry_guidance_accounts_for_possible_commit(
    operation: str, expected_retry: str
):
    request = httpx.Request("POST", "https://mettle.sh/api/session")
    response = httpx.Response(503, request=request)
    error = httpx.HTTPStatusError("unavailable", request=request, response=response)

    result = mcp_server.failure(operation, error)

    assert _assert_valid(result)["error"]["retry"] == expected_retry


@pytest.mark.asyncio
@pytest.mark.parametrize("round_num", [True, 0, 6, "1", "1/feedback"])
async def test_round_paths_reject_non_integer_and_out_of_range_values(round_num):
    with patch.object(mcp_server, "api_call", AsyncMock()) as call:
        result = await mcp_server.call_tool(
            "mettle_get_round_feedback",
            {"session_id": "ses_a", "round_num": round_num},
        )

    envelope = _assert_valid(result)
    assert result.is_error is True
    assert envelope["error"]["code"] == "invalid_request"
    call.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_inspection_supports_both_authority_profiles():
    quick = {
        "session_id": "ses_q",
        "status": "in_progress",
        "completed_challenges": 1,
        "total_challenges": 3,
    }
    authenticated = {
        "session_id": "ses_a",
        "status": "challenges_generated",
        "suites": ["native"],
        "suites_completed": [],
        "current_round": 0,
    }
    call = AsyncMock(side_effect=[quick, authenticated])
    mcp_server._remember_session_token("ses_q", "tok_q")
    with patch.object(mcp_server, "api_call", call):
        quick_result = await mcp_server.call_tool(
            "mettle_get_session", {"session_id": "ses_q", "profile": "quick"}
        )
        auth_result = await mcp_server.call_tool(
            "mettle_get_session", {"session_id": "ses_a"}
        )

    quick_envelope = _assert_valid(quick_result)
    auth_envelope = _assert_valid(auth_result)
    assert quick_envelope["snapshot"]["profile"] == "quick"
    assert auth_envelope["snapshot"]["profile"] == "authenticated"
    assert auth_envelope["actions"][0]["operation"] == "mettle_verify_suite"


@pytest.mark.asyncio
async def test_cancel_returns_terminal_snapshot_and_receipt():
    delete = AsyncMock(return_value=_response({}, 204))
    with patch.object(mcp_server, "API_KEY", "mtl_test"):
        with patch.object(mcp_server.http_client, "delete", delete):
            result = await mcp_server.call_tool(
                "mettle_cancel_session", {"session_id": "ses_a"}
            )

    envelope = _assert_valid(result)
    assert envelope["snapshot"]["terminal"] is True
    assert envelope["snapshot"]["status"] == "cancelled"
    assert envelope["receipt"]["accepted"] is True
    assert envelope["actions"] == []


@pytest.mark.asyncio
async def test_first_contact_multi_round_flow_uses_only_schemas_and_structured_state():
    tools = {tool.name: tool for tool in await mcp_server.list_tools()}
    calls = AsyncMock(
        side_effect=[
            {
                "session_id": "ses_novel",
                "status": "challenges_generated",
                "suites": ["novel-reasoning"],
                "suites_completed": [],
                "current_round": 0,
                "challenges": {"novel-reasoning": {"round": 1}},
                "time_budget_ms": 90000,
            },
            {
                "session_id": "ses_novel",
                "status": "in_progress",
                "suites": ["novel-reasoning"],
                "suites_completed": [],
                "current_round": 1,
            },
            {
                "round_num": 1,
                "accuracy": 0.75,
                "errors": [],
                "feedback": {"accuracy": 0.75},
                "time_remaining_ms": 60000,
                "next_round_data": {"round": 2},
            },
            {
                "round": 1,
                "accuracy": 0.75,
                "response_time_ms": 1000,
            },
            {
                "session_id": "ses_novel",
                "status": "in_progress",
                "suites": ["novel-reasoning"],
                "suites_completed": [],
                "current_round": 1,
            },
        ]
    )
    start_args = {"suites": ["novel-reasoning"], "difficulty": "easy"}
    jsonschema.validate(start_args, tools["mettle_start_v2_session"].input_schema)
    with patch.object(mcp_server, "api_call", calls):
        started = await mcp_server.call_tool("mettle_start_v2_session", start_args)
        started_envelope = _assert_valid(started)
        next_action = started_envelope["actions"][0]
        assert next_action["operation"] == "mettle_submit_round"

        round_args = {**next_action["arguments"], "answers": {"pattern": "A"}}
        jsonschema.validate(round_args, tools["mettle_submit_round"].input_schema)
        submitted = await mcp_server.call_tool("mettle_submit_round", round_args)
        submitted_envelope = _assert_valid(submitted)
        assert submitted_envelope["snapshot"]["current_round"] == 1
        assert submitted_envelope["actions"][0]["arguments"]["round_num"] == 2

        feedback_args = {"session_id": "ses_novel", "round_num": 1}
        jsonschema.validate(
            feedback_args, tools["mettle_get_round_feedback"].input_schema
        )
        feedback = await mcp_server.call_tool(
            "mettle_get_round_feedback", feedback_args
        )
        assert _assert_valid(feedback)["data"]["accuracy"] == 0.75


@pytest.mark.asyncio
async def test_authenticated_single_and_multi_round_mcp_journeys_reach_real_routes():
    """Run the adapter against FastAPI and SessionManager, with only Redis faked."""
    redis = FakeRedis()
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_authenticated_user] = lambda: _make_mock_user()

    async def manager() -> SessionManager:
        return SessionManager(redis)

    app.dependency_overrides[get_session_manager] = manager
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        with patch.object(mcp_server, "http_client", client):
            with patch.object(mcp_server, "API_URL", "http://testserver/api"):
                with patch.object(mcp_server, "API_KEY", "mtl_test"):
                    single = await mcp_server.call_tool(
                        "mettle_start_v2_session",
                        {"suites": ["native"], "difficulty": "easy"},
                    )
                    single_id = _assert_valid(single)["snapshot"]["session_id"]
                    submitted = await mcp_server.call_tool(
                        "mettle_verify_suite",
                        {"session_id": single_id, "suite": "native", "answers": {}},
                    )
                    assert _assert_valid(submitted)["snapshot"]["terminal"] is True
                    single_result = await mcp_server.call_tool(
                        "mettle_get_v2_result",
                        {"session_id": single_id, "include_vcp": False},
                    )
                    assert _assert_valid(single_result)["outcome"] == "succeeded"

                    novel = await mcp_server.call_tool(
                        "mettle_start_v2_session",
                        {"suites": ["novel-reasoning"], "difficulty": "easy"},
                    )
                    novel_envelope = _assert_valid(novel)
                    novel_id = novel_envelope["snapshot"]["session_id"]
                    challenge_names = novel_envelope["data"]["challenges"][
                        "novel-reasoning"
                    ]["challenges"]
                    complete_answer_shape: dict[str, dict[str, object]] = {
                        name: {} for name in challenge_names
                    }
                    for round_num in (1, 2):
                        round_result = await mcp_server.call_tool(
                            "mettle_submit_round",
                            {
                                "session_id": novel_id,
                                "round_num": round_num,
                                "answers": complete_answer_shape,
                            },
                        )
                        round_envelope = _assert_valid(round_result)
                        assert round_envelope["outcome"] == "succeeded", json.dumps(
                            round_envelope
                        )
                        assert round_envelope["data"]["round_num"] == round_num
                    assert round_envelope["snapshot"]["terminal"] is True
                    novel_result = await mcp_server.call_tool(
                        "mettle_get_v2_result",
                        {"session_id": novel_id, "include_vcp": False},
                    )
                    assert _assert_valid(novel_result)["outcome"] == "succeeded"
