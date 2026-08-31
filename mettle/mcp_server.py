#!/usr/bin/env python3
"""
METTLE MCP Server

Model Context Protocol server that allows Becoming Minds to screen themselves
through METTLE challenges. Provides tools for starting sessions, answering
challenges, and retrieving results.

Usage:
    mettle-mcp

Configuration (environment variables):
    METTLE_API_URL - Base URL for METTLE API (default: https://mettle.sh/api)
    METTLE_API_KEY - Bearer key for the v2 suites/tiers/attestation endpoints.
                     METTLE_API_KEYS (comma-separated) is accepted as a fallback;
                     the first entry is used. Never hardcode a key here.
"""

import argparse
import asyncio
import json as jsonlib
import os
import re
import time
from typing import Any

import httpx
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ContentBlock,
    ListToolsResult,
    PaginatedRequestParams,
    Tool,
    ToolAnnotations,
)
from mettle import __version__
from mettle.mcp_context import caller_api_key, caller_principal, http_request_active
from mettle.mcp_contract import (
    CONTROL_OUTPUT_SCHEMA,
    ToolResponse,
    failure,
    normalize_status,
    receipt,
    session_actions,
    success,
)

# Configuration
API_URL = os.getenv("METTLE_API_URL", "https://mettle.sh/api")
# The v2 suites/tiers/attestation endpoints (/api/mettle/*) require a Bearer key; the
# operator supplies the client's key here (the server validates it against its own
# METTLE_API_KEYS). The MVP /session/* endpoints use a *different*, per-session
# credential: /session/start mints a `session_token` which every subsequent call on
# that session must present as `X-Session-Token` (the API returns 401 otherwise).
API_KEY = os.getenv("METTLE_API_KEY") or next(
    (k.strip() for k in os.getenv("METTLE_API_KEYS", "").split(",") if k.strip()), None
)

# HTTP client for API calls
http_client = httpx.AsyncClient(timeout=30.0)


# === Helper Functions ===


async def api_call(
    endpoint: str,
    method: str = "GET",
    json: dict | None = None,
    params: dict | None = None,
    auth: bool = False,
    session_token: str | None = None,
) -> Any:
    """Make an API call to METTLE.

    Two independent credentials exist:

    * ``auth=True`` -- the v2 ``/api/mettle/*`` endpoints, which require an operator
      Bearer key (``METTLE_API_KEY``).
    * ``session_token`` -- the MVP ``/session/*`` endpoints, which require the
      per-session token minted by ``/session/start``, sent as ``X-Session-Token``.
    """
    url = f"{API_URL}{endpoint}"
    headers: dict[str, str] = {}
    request_key = caller_api_key.get()
    if http_request_active.get() and not request_key:
        raise RuntimeError("Authenticated MCP caller context is unavailable")
    effective_api_key = request_key or API_KEY
    if auth:
        if not effective_api_key:
            raise RuntimeError(
                "METTLE_API_KEY is required for the METTLE v2 (suites/tiers/attestation) endpoints"
            )
        headers["Authorization"] = f"Bearer {effective_api_key}"
    elif effective_api_key:
        # Legacy session creation uses this key for caller-specific daily quota.
        # HTTP mode supplies the authenticated caller's key, never a shared
        # process credential.
        headers["X-API-Key"] = effective_api_key
    if session_token:
        headers["X-Session-Token"] = session_token

    if method == "GET":
        response = await http_client.get(url, headers=headers, params=params)
    elif method == "POST":
        response = await http_client.post(
            url, json=json, headers=headers, params=params
        )
    elif method == "DELETE":
        response = await http_client.delete(url, headers=headers, params=params)
    else:
        raise ValueError("Unsupported HTTP method")

    response.raise_for_status()
    return (
        {} if response.status_code == 204 or not response.content else response.json()
    )


_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SESSION_TOKEN_TTL_SECONDS = 3600
_MAX_SESSION_TOKENS = 10_000
_session_tokens: dict[tuple[str, str], tuple[str, float]] = {}


def _safe_id(value: object, what: str) -> str:
    """Validate an identifier before it is interpolated into a request URL.

    Agent-supplied ``session_id`` / ``suite`` values flow into credentialed request
    paths; a value containing ``/``, ``?``, ``#`` or ``..`` could redirect the request
    to another path on the host. Reject anything but a strict ``[A-Za-z0-9_-]{1,64}``
    token so path injection can't reach the URL.
    """
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValueError(f"invalid {what}: must match [A-Za-z0-9_-] (1-64 chars)")
    return value


def _round_number(value: object) -> int:
    """Validate a round before interpolating it into an authenticated URL."""
    if type(value) is not int or not 1 <= value <= 5:
        raise ValueError("invalid round_num: must be an integer from 1 to 5")
    return value


def _session_token_key(session_id: str) -> tuple[str, str]:
    return caller_principal.get(), _safe_id(session_id, "session_id")


def _remember_session_token(session_id: str, token: str) -> None:
    """Keep a per-caller session credential outside model-visible content."""
    if not isinstance(token, str) or not token:
        raise ValueError("Session authority did not return a usable token")
    now = time.monotonic()
    expired = [key for key, (_, expiry) in _session_tokens.items() if expiry <= now]
    for key in expired:
        _session_tokens.pop(key, None)
    if len(_session_tokens) >= _MAX_SESSION_TOKENS:
        oldest = min(_session_tokens, key=lambda key: _session_tokens[key][1])
        _session_tokens.pop(oldest, None)
    _session_tokens[_session_token_key(session_id)] = (
        token,
        now + _SESSION_TOKEN_TTL_SECONDS,
    )


def _get_session_token(session_id: str, *, consume: bool = False) -> str:
    key = _session_token_key(session_id)
    record = _session_tokens.pop(key, None) if consume else _session_tokens.get(key)
    if record is None or record[1] <= time.monotonic():
        _session_tokens.pop(key, None)
        raise ValueError("Unknown or expired session for this MCP caller")
    return record[0]


# === MCP Tools ===


def _annotations(
    *,
    read_only: bool,
    destructive: bool,
    idempotent: bool,
) -> ToolAnnotations:
    """Declare control effects accurately for MCP hosts."""
    return ToolAnnotations(
        read_only_hint=read_only,
        destructive_hint=destructive,
        idempotent_hint=idempotent,
        open_world_hint=True,
    )


async def list_tools() -> list[Tool]:
    """List the compatibility tools and additive control-v1 operations."""
    output = CONTROL_OUTPUT_SCHEMA
    return [
        Tool(
            name="mettle_start_session",
            description=(
                "Start a quick METTLE session and return the first challenge. "
                "The session bearer stays in a caller-isolated vault."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "difficulty": {
                        "type": "string",
                        "enum": ["basic", "full"],
                        "default": "basic",
                    },
                    "entity_id": {"type": "string", "maxLength": 128},
                },
            },
            output_schema=output,
            annotations=_annotations(
                read_only=False, destructive=False, idempotent=False
            ),
        ),
        Tool(
            name="mettle_answer_challenge",
            description=(
                "Submit the current quick challenge answer and return its receipt, "
                "the next snapshot, and valid next actions."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "session_id": {"type": "string", "minLength": 1, "maxLength": 64},
                    "challenge_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 64,
                    },
                    "answer": {"type": "string", "maxLength": 1024},
                },
                "required": ["session_id", "challenge_id", "answer"],
            },
            output_schema=output,
            annotations=_annotations(
                read_only=False, destructive=True, idempotent=False
            ),
        ),
        Tool(
            name="mettle_get_result",
            description=(
                "Read a completed quick-session result. Reads are repeatable while "
                "the hidden caller capability remains available."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "session_id": {"type": "string", "minLength": 1, "maxLength": 64}
                },
                "required": ["session_id"],
            },
            output_schema=output,
            annotations=_annotations(
                read_only=False, destructive=False, idempotent=True
            ),
        ),
        Tool(
            name="mettle_list_suites",
            description="List authenticated METTLE suites and their availability.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
            output_schema=output,
            annotations=_annotations(
                read_only=True, destructive=False, idempotent=True
            ),
        ),
        Tool(
            name="mettle_start_v2_session",
            description=(
                "Start an authenticated suite session. Responses to llm-dynamic "
                "leave METTLE only after explicit acknowledgement."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "suites": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 12,
                        "default": ["all"],
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["easy", "standard", "hard"],
                        "default": "standard",
                    },
                    "entity_id": {"type": "string", "maxLength": 256},
                    "vcp_token": {"type": "string", "maxLength": 32768},
                    "allow_third_party_llm": {"type": "boolean", "default": False},
                },
            },
            output_schema=output,
            annotations=_annotations(
                read_only=False, destructive=False, idempotent=False
            ),
        ),
        Tool(
            name="mettle_verify_suite",
            description=(
                "Submit one single-shot suite and return the result plus current "
                "session snapshot. Multi-round suites use mettle_submit_round."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "session_id": {"type": "string", "minLength": 1, "maxLength": 64},
                    "suite": {"type": "string", "minLength": 1, "maxLength": 64},
                    "answers": {"type": "object"},
                },
                "required": ["session_id", "suite", "answers"],
            },
            output_schema=output,
            annotations=_annotations(
                read_only=False, destructive=True, idempotent=False
            ),
        ),
        Tool(
            name="mettle_get_v2_result",
            description=(
                "Read the terminal authenticated result and optional signed "
                "credential or evidence receipt."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "session_id": {"type": "string", "minLength": 1, "maxLength": 64},
                    "include_vcp": {"type": "boolean", "default": True},
                },
                "required": ["session_id"],
            },
            output_schema=output,
            annotations=_annotations(
                read_only=False, destructive=False, idempotent=True
            ),
        ),
        Tool(
            name="mettle_get_session",
            description=(
                "Inspect a quick or authenticated session using its non-secret "
                "handle. Defaults to the authenticated profile."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "session_id": {"type": "string", "minLength": 1, "maxLength": 64},
                    "profile": {
                        "type": "string",
                        "enum": ["quick", "authenticated"],
                        "default": "authenticated",
                    },
                },
                "required": ["session_id"],
            },
            output_schema=output,
            annotations=_annotations(
                read_only=False, destructive=False, idempotent=True
            ),
        ),
        Tool(
            name="mettle_cancel_session",
            description=(
                "Cancel an active authenticated session and return its terminal "
                "snapshot. Quick-session cancellation is not currently supported."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "session_id": {"type": "string", "minLength": 1, "maxLength": 64}
                },
                "required": ["session_id"],
            },
            output_schema=output,
            annotations=_annotations(
                read_only=False, destructive=True, idempotent=False
            ),
        ),
        Tool(
            name="mettle_submit_round",
            description=(
                "Submit one novel-reasoning round and return bounded feedback, the "
                "next round data, and the current session snapshot."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "session_id": {"type": "string", "minLength": 1, "maxLength": 64},
                    "round_num": {"type": "integer", "minimum": 1, "maximum": 5},
                    "answers": {"type": "object"},
                },
                "required": ["session_id", "round_num", "answers"],
            },
            output_schema=output,
            annotations=_annotations(
                read_only=False, destructive=True, idempotent=False
            ),
        ),
        Tool(
            name="mettle_get_round_feedback",
            description="Read the recorded feedback for one completed novel-reasoning round.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "session_id": {"type": "string", "minLength": 1, "maxLength": 64},
                    "round_num": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["session_id", "round_num"],
            },
            output_schema=output,
            annotations=_annotations(
                read_only=True, destructive=False, idempotent=True
            ),
        ),
    ]


def _quick_snapshot(
    session_id: str,
    *,
    status: object,
    current_challenge: dict[str, Any] | None = None,
    **facts: Any,
) -> dict[str, Any]:
    control_status = normalize_status(status)
    return {
        "session_id": session_id,
        "profile": "quick",
        "status": control_status,
        "terminal": control_status in {"completed", "expired", "cancelled"},
        "current_challenge": current_challenge,
        **facts,
    }


def _authenticated_snapshot(
    data: dict[str, Any], session_id: str | None = None
) -> dict[str, Any]:
    control_status = normalize_status(data.get("status", "ready"))
    return {
        "session_id": session_id or str(data["session_id"]),
        "profile": "authenticated",
        "status": control_status,
        "terminal": control_status in {"completed", "expired", "cancelled"},
        "suites": list(data.get("suites", [])),
        "suites_completed": list(data.get("suites_completed", [])),
        "current_round": data.get("current_round"),
        "created_at": data.get("created_at"),
        "expires_at": data.get("expires_at"),
        "elapsed_ms": data.get("elapsed_ms"),
        "challenges": data.get("challenges"),
    }


async def _authenticated_status(session_id: str) -> dict[str, Any]:
    return await api_call(f"/mettle/sessions/{session_id}", auth=True)


async def call_tool(name: str, arguments: dict[str, Any]) -> ToolResponse:
    """Handle one tool call with compatibility text and control-v1 content."""
    try:
        if name == "mettle_start_session":
            data = await api_call(
                "/session/start",
                "POST",
                {
                    "difficulty": arguments.get("difficulty", "basic"),
                    "entity_id": arguments.get("entity_id"),
                },
            )
            session_id = _safe_id(data["session_id"], "session_id")
            _remember_session_token(session_id, data["session_token"])
            challenge = data["current_challenge"]
            snapshot = _quick_snapshot(
                session_id,
                status="ready",
                current_challenge=challenge,
                difficulty=data["difficulty"],
                total_challenges=data["total_challenges"],
                completed_challenges=0,
            )
            text = (
                "METTLE session started.\n"
                f"Session ID: {session_id}\n"
                f"Difficulty: {data['difficulty']}\n"
                f"Total challenges: {data['total_challenges']}\n"
                f"First challenge: {challenge['prompt']}\n"
                f"Challenge ID: {challenge['id']}\n"
                f"Time limit: {challenge['time_limit_ms']}ms"
            )
            public_data = {
                key: value for key, value in data.items() if key != "session_token"
            }
            return success(
                name,
                text,
                data=public_data,
                snapshot=snapshot,
                actions=session_actions(snapshot),
                mutation_receipt=receipt(name, session_id=session_id),
            )

        if name == "mettle_answer_challenge":
            session_id = _safe_id(arguments["session_id"], "session_id")
            data = await api_call(
                "/session/answer",
                "POST",
                {
                    "session_id": session_id,
                    "challenge_id": arguments["challenge_id"],
                    "answer": arguments["answer"],
                },
                session_token=_get_session_token(session_id),
            )
            result = data["result"]
            complete = bool(data["session_complete"])
            snapshot = _quick_snapshot(
                session_id,
                status="completed" if complete else "in_progress",
                current_challenge=data.get("next_challenge"),
                challenges_remaining=data.get("challenges_remaining", 0),
            )
            verdict = "PASSED" if result["passed"] else "FAILED"
            text = (
                f"Challenge result: {verdict}\n"
                f"Response time: {result['response_time_ms']}ms "
                f"(limit: {result['time_limit_ms']}ms)\n"
                f"Challenges remaining: {data.get('challenges_remaining', 0)}"
            )
            if data.get("next_challenge"):
                text += (
                    f"\nNext challenge: {data['next_challenge']['prompt']}"
                    f"\nChallenge ID: {data['next_challenge']['id']}"
                )
            return success(
                name,
                text,
                data=data,
                snapshot=snapshot,
                actions=session_actions(snapshot),
                mutation_receipt=receipt(
                    name,
                    session_id=session_id,
                    challenge_id=arguments["challenge_id"],
                    accepted=True,
                ),
            )

        if name == "mettle_get_result":
            session_id = _safe_id(arguments["session_id"], "session_id")
            data = await api_call(
                f"/session/{session_id}/result",
                session_token=_get_session_token(session_id),
            )
            snapshot = _quick_snapshot(
                session_id,
                status="completed",
                verified=bool(data.get("verified")),
                passed=data.get("passed"),
                total=data.get("total"),
                credential_eligible=data.get("credential_eligible"),
            )
            verdict = "VERIFIED" if data["verified"] else "NOT VERIFIED"
            lines = [
                "METTLE Verification Result",
                f"Status: {verdict}",
                f"Passed: {data['passed']}/{data['total']} "
                f"({data['pass_rate'] * 100:.0f}%)",
            ]
            if data.get("badge"):
                lines.append(f"Badge: {data['badge']}")
            if data.get("entity_id"):
                lines.append(f"Entity: {data['entity_id']}")
            for item in data.get("results", []):
                item_status = "PASS" if item["passed"] else "FAIL"
                lines.append(f"{item['challenge_type']}: {item_status}")
            return success(
                name,
                "\n".join(lines),
                data=data,
                snapshot=snapshot,
                actions=session_actions(snapshot),
            )

        if name == "mettle_list_suites":
            data = await api_call("/mettle/suites", auth=True)
            lines = ["METTLE authenticated verification suites:"]
            for suite in data:
                flags = []
                if suite.get("is_multi_round"):
                    flags.append("multi-round")
                if not suite.get("available", True):
                    flags.append("unavailable")
                suffix = f" [{', '.join(flags)}]" if flags else ""
                lines.append(
                    f"{suite.get('suite_number')}. {suite.get('name')} "
                    f"({suite.get('display_name')}){suffix}"
                )
            return success(name, "\n".join(lines), data=data)

        if name == "mettle_start_v2_session":
            payload: dict[str, Any] = {
                "suites": arguments.get("suites", ["all"]),
                "difficulty": arguments.get("difficulty", "standard"),
            }
            for optional in ("entity_id", "vcp_token"):
                if arguments.get(optional):
                    payload[optional] = arguments[optional]
            if arguments.get("allow_third_party_llm") is True:
                payload["allow_third_party_llm"] = True
            data = await api_call("/mettle/sessions", "POST", payload, auth=True)
            session_id = _safe_id(data["session_id"], "session_id")
            snapshot = _authenticated_snapshot(data, session_id)
            text = (
                "METTLE authenticated session started.\n"
                f"Session ID: {data['session_id']}\n"
                f"Suites: {', '.join(data.get('suites', []))}\n"
                f"Time budget: {data.get('time_budget_ms')}ms"
            )
            return success(
                name,
                text,
                data=data,
                snapshot=snapshot,
                actions=session_actions(snapshot),
                mutation_receipt=receipt(name, session_id=session_id),
            )

        if name == "mettle_verify_suite":
            session_id = _safe_id(arguments["session_id"], "session_id")
            suite = _safe_id(arguments["suite"], "suite")
            status_data = await _authenticated_status(session_id)
            data = await api_call(
                f"/mettle/sessions/{session_id}/verify",
                "POST",
                {"suite": suite, "answers": arguments["answers"]},
                auth=True,
            )
            completed = list(status_data.get("suites_completed", []))
            if suite not in completed:
                completed.append(suite)
            status_data = {
                **status_data,
                "suites_completed": completed,
                "status": (
                    "completed"
                    if set(completed) == set(status_data.get("suites", []))
                    else "in_progress"
                ),
            }
            snapshot = _authenticated_snapshot(status_data, session_id)
            verdict = "PASSED" if data.get("passed") else "FAILED"
            text = f"Suite '{data.get('suite')}': {verdict}\nScore: {data.get('score')}"
            if data.get("details"):
                text += f"\nDetails: {jsonlib.dumps(data['details'])}"
            return success(
                name,
                text,
                data=data,
                snapshot=snapshot,
                actions=session_actions(snapshot),
                mutation_receipt=receipt(name, session_id=session_id, suite=suite),
            )

        if name == "mettle_get_v2_result":
            session_id = _safe_id(arguments["session_id"], "session_id")
            include_vcp = arguments.get("include_vcp", True)
            data = await api_call(
                f"/mettle/sessions/{session_id}/result",
                params={"include_vcp": str(include_vcp).lower()},
                auth=True,
            )
            snapshot = _authenticated_snapshot(data, session_id)
            verdict = "PASSED" if data.get("overall_passed") else "NOT PASSED"
            text = (
                "METTLE authenticated result\n"
                f"Status: {data.get('status')}\n"
                f"Overall: {verdict}\n"
                f"Tier: {data.get('tier') or 'none'}"
            )
            if data.get("vcp_attestation"):
                text += (
                    "\nVCP attestation: "
                    f"{jsonlib.dumps(data['vcp_attestation'], indent=2)}"
                )
            if data.get("governance_attestation"):
                text += (
                    "\nGovernance attestation: "
                    f"framework={data['governance_attestation'].get('framework')}"
                )
            return success(
                name,
                text,
                data=data,
                snapshot=snapshot,
                actions=session_actions(snapshot),
            )

        if name == "mettle_get_session":
            session_id = _safe_id(arguments["session_id"], "session_id")
            profile = arguments.get("profile", "authenticated")
            if profile == "quick":
                data = await api_call(
                    f"/session/{session_id}",
                    session_token=_get_session_token(session_id),
                )
                snapshot = _quick_snapshot(
                    session_id,
                    status=data.get("status"),
                    completed_challenges=data.get("completed_challenges"),
                    total_challenges=data.get("total_challenges"),
                )
            elif profile == "authenticated":
                data = await _authenticated_status(session_id)
                snapshot = _authenticated_snapshot(data, session_id)
            else:
                raise ValueError("Invalid session profile")
            return success(
                name,
                f"Session {session_id}: {snapshot['status']}",
                data=data,
                snapshot=snapshot,
                actions=session_actions(snapshot),
            )

        if name == "mettle_cancel_session":
            session_id = _safe_id(arguments["session_id"], "session_id")
            await api_call(
                f"/mettle/sessions/{session_id}",
                "DELETE",
                auth=True,
            )
            data = {
                "session_id": session_id,
                "status": "cancelled",
                "suites": [],
                "suites_completed": [],
                "current_round": None,
            }
            snapshot = _authenticated_snapshot(data, session_id)
            return success(
                name,
                f"Session {session_id}: {snapshot['status']}",
                data=data,
                snapshot=snapshot,
                actions=session_actions(snapshot),
                mutation_receipt=receipt(name, session_id=session_id),
            )

        if name == "mettle_submit_round":
            session_id = _safe_id(arguments["session_id"], "session_id")
            round_num = _round_number(arguments["round_num"])
            status_data = await _authenticated_status(session_id)
            data = await api_call(
                f"/mettle/sessions/{session_id}/rounds/{round_num}/answer",
                "POST",
                {"answers": arguments["answers"]},
                auth=True,
            )
            completed = list(status_data.get("suites_completed", []))
            if (
                data.get("next_round_data") is None
                and "novel-reasoning" not in completed
            ):
                completed.append("novel-reasoning")
            status_data = {
                **status_data,
                "current_round": round_num,
                "suites_completed": completed,
                "status": (
                    "completed"
                    if set(completed) == set(status_data.get("suites", []))
                    else "in_progress"
                ),
            }
            snapshot = _authenticated_snapshot(status_data, session_id)
            text = (
                f"Round {data.get('round_num')}: accuracy {data.get('accuracy')}\n"
                f"Time remaining: {data.get('time_remaining_ms')}ms"
            )
            return success(
                name,
                text,
                data=data,
                snapshot=snapshot,
                actions=session_actions(snapshot),
                mutation_receipt=receipt(
                    name, session_id=session_id, round_num=round_num
                ),
            )

        if name == "mettle_get_round_feedback":
            session_id = _safe_id(arguments["session_id"], "session_id")
            round_num = _round_number(arguments["round_num"])
            data = await api_call(
                f"/mettle/sessions/{session_id}/rounds/{round_num}/feedback",
                auth=True,
            )
            status_data = await _authenticated_status(session_id)
            snapshot = _authenticated_snapshot(status_data, session_id)
            return success(
                name,
                f"Feedback for round {round_num} is available.",
                data=data,
                snapshot=snapshot,
                actions=session_actions(snapshot),
            )

        raise ValueError("Unknown MCP tool")
    except Exception as exc:
        return failure(name, exc)


async def _handle_list_tools(
    _context: ServerRequestContext[Any], _params: PaginatedRequestParams | None
) -> ListToolsResult:
    """Adapt METTLE's transport-independent listing to MCP 2's low-level API."""
    return ListToolsResult(tools=await list_tools())


async def _handle_call_tool(
    _context: ServerRequestContext[Any], params: CallToolRequestParams
) -> CallToolResult:
    """Adapt an MCP 2 request model to the transport-independent tool handler."""
    response = await call_tool(params.name, params.arguments or {})
    content: list[ContentBlock] = list(response)
    return CallToolResult(
        content=content,
        structured_content=response.structured_content,
        is_error=response.is_error,
    )


# MCP 2 registers low-level handlers in the constructor. Keeping the core
# ``list_tools`` and ``call_tool`` functions transport-independent also makes
# their behaviour straightforward to unit test.
server: Server[Any] = Server(
    "mettle",
    version=__version__,
    on_list_tools=_handle_list_tools,
    on_call_tool=_handle_call_tool,
)


async def run_server() -> None:  # pragma: no cover
    """Serve the MCP protocol over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the ``mettle-mcp`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="mettle-mcp",
        description="METTLE MCP server for behavioral screening and credential workflows.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport to serve on (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "HTTP bind address (default: 127.0.0.1). Binding a non-loopback "
            "address requires METTLE_MCP_ALLOW_INSECURE_HTTP=true."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="HTTP port. Defaults to $PORT if set, else 8080.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point (``mettle-mcp``)."""
    args = build_arg_parser().parse_args(argv)

    if args.transport == "http":
        # Imported lazily: the http extras (starlette/uvicorn) are only needed
        # on this path, and a stdio-only environment must not pay for them.
        from mettle._http import run_http

        run_http(server, args.host, args.port, list_tools)
        return

    asyncio.run(run_server())


if __name__ == "__main__":  # pragma: no cover
    main()
