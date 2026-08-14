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
import json
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
    TextContent,
    Tool,
)
from mettle import __version__
from mettle.mcp_context import caller_api_key, caller_principal, http_request_active

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
) -> dict:
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
    else:
        response = await http_client.post(
            url, json=json, headers=headers, params=params
        )

    response.raise_for_status()
    return response.json()


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


async def list_tools() -> list[Tool]:
    """List available METTLE tools."""
    return [
        Tool(
            name="mettle_start_session",
            description=(
                "Start a METTLE verification session for a Becoming Mind. "
                "Returns the first challenge to solve. Use difficulty='basic' for 3 challenges "
                "(relaxed timing) or 'full' for 5 challenges (strict timing)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "difficulty": {
                        "type": "string",
                        "enum": ["basic", "full"],
                        "description": "Verification difficulty level",
                        "default": "basic",
                    },
                    "entity_id": {
                        "type": "string",
                        "description": "Optional identifier for this Becoming Mind",
                    },
                },
            },
        ),
        Tool(
            name="mettle_answer_challenge",
            description=(
                "Submit an answer to the current METTLE challenge. "
                "Returns the verification result and next challenge (if any)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID from mettle_start_session",
                    },
                    "challenge_id": {
                        "type": "string",
                        "description": "Challenge ID to answer",
                    },
                    "answer": {
                        "type": "string",
                        "description": "Your answer to the challenge",
                    },
                },
                "required": ["session_id", "challenge_id", "answer"],
            },
        ),
        Tool(
            name="mettle_get_result",
            description=(
                "Get the final verification result for a completed METTLE session. "
                "Shows whether you passed (80% threshold) and your verification badge."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to get results for",
                    },
                },
                "required": ["session_id"],
            },
        ),
        # --- v2 suites / tiers / VCP attestation (require METTLE_API_KEY) ---
        Tool(
            name="mettle_list_suites",
            description=(
                "List the METTLE v2 verification suites (the harder, tiered credential path). "
                "Each suite probes a capability dimension; passing sets earn a tier "
                "(bronze/silver/gold/platinum). Requires METTLE_API_KEY."
            ),
            input_schema={"type": "object", "properties": {}},
        ),
        Tool(
            name="mettle_start_v2_session",
            description=(
                "Start a METTLE v2 verification session over one or more suites and receive the "
                "challenge data (never the answers). Answer each suite with mettle_verify_suite. "
                "Requires METTLE_API_KEY."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "suites": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Suite names, or ['all']",
                        "default": ["all"],
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["easy", "standard", "hard"],
                        "description": "Difficulty level",
                        "default": "standard",
                    },
                    "entity_id": {
                        "type": "string",
                        "description": "Optional identifier for this Becoming Mind",
                    },
                    "vcp_token": {
                        "type": "string",
                        "description": "Optional CSM-1 VCP token (enhanced Suite 9 / governance attestation)",
                    },
                    "allow_third_party_llm": {
                        "type": "boolean",
                        "description": (
                            "Explicitly acknowledge that llm-dynamic responses are "
                            "sent to Anthropic for evaluation"
                        ),
                        "default": False,
                    },
                },
            },
        ),
        Tool(
            name="mettle_verify_suite",
            description=(
                "Submit your answers for one single-shot suite in a v2 session and get pass/score. "
                "Requires METTLE_API_KEY."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID from mettle_start_v2_session",
                    },
                    "suite": {"type": "string", "description": "Suite name to verify"},
                    "answers": {
                        "type": "object",
                        "description": "Suite-specific answers (your responses)",
                    },
                },
                "required": ["session_id", "suite", "answers"],
            },
        ),
        Tool(
            name="mettle_get_v2_result",
            description=(
                "Get the final v2 result for a session: overall pass, the earned tier, and (by "
                "default) the signed VCP attestation you can present as a credential. Requires "
                "METTLE_API_KEY."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to get results for",
                    },
                    "include_vcp": {
                        "type": "boolean",
                        "description": "Include the VCP-compatible attestation (tier + signature)",
                        "default": True,
                    },
                },
                "required": ["session_id"],
            },
        ),
    ]


async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    if name == "mettle_start_session":
        try:
            difficulty = arguments.get("difficulty", "basic")
            entity_id = arguments.get("entity_id")

            data = await api_call(
                "/session/start",
                "POST",
                {"difficulty": difficulty, "entity_id": entity_id},
            )

            challenge = data["current_challenge"]
            _remember_session_token(data["session_id"], data["session_token"])
            return [
                TextContent(
                    type="text",
                    text=(
                        f"METTLE session started!\n\n"
                        f"Session ID: {data['session_id']}\n"
                        f"Difficulty: {data['difficulty']}\n"
                        f"Total challenges: {data['total_challenges']}\n\n"
                        f"First Challenge:\n"
                        f"  ID: {challenge['id']}\n"
                        f"  Type: {challenge['type']}\n"
                        f"  Prompt: {challenge['prompt']}\n"
                        f"  Time limit: {challenge['time_limit_ms']}ms\n\n"
                        f"Use mettle_answer_challenge to submit your answer."
                    ),
                )
            ]
        except httpx.HTTPStatusError as e:
            return [
                TextContent(
                    type="text", text=f"Error starting session: {e.response.text}"
                )
            ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "mettle_answer_challenge":
        try:
            data = await api_call(
                "/session/answer",
                "POST",
                {
                    "session_id": arguments["session_id"],
                    "challenge_id": arguments["challenge_id"],
                    "answer": arguments["answer"],
                },
                session_token=_get_session_token(arguments["session_id"]),
            )

            result = data["result"]
            passed_text = "PASSED" if result["passed"] else "FAILED"

            response_text = (
                f"Challenge Result: {passed_text}\n"
                f"Response time: {result['response_time_ms']}ms (limit: {result['time_limit_ms']}ms)\n"
            )

            if data["session_complete"]:
                response_text += (
                    "\nSession complete! Challenges remaining: 0\n"
                    "Use mettle_get_result to see your final verification result."
                )
            else:
                next_challenge = data["next_challenge"]
                response_text += (
                    f"\nChallenges remaining: {data['challenges_remaining']}\n\n"
                    f"Next Challenge:\n"
                    f"  ID: {next_challenge['id']}\n"
                    f"  Type: {next_challenge['type']}\n"
                    f"  Prompt: {next_challenge['prompt']}\n"
                    f"  Time limit: {next_challenge['time_limit_ms']}ms"
                )

            return [TextContent(type="text", text=response_text)]
        except httpx.HTTPStatusError as e:
            return [
                TextContent(
                    type="text", text=f"Error submitting answer: {e.response.text}"
                )
            ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "mettle_get_result":
        try:
            session_id = _safe_id(arguments["session_id"], "session_id")
            data = await api_call(
                f"/session/{session_id}/result",
                session_token=_get_session_token(session_id),
            )

            verified_text = "VERIFIED" if data["verified"] else "NOT VERIFIED"

            response_text = (
                f"METTLE Verification Result\n"
                f"{'=' * 30}\n\n"
                f"Status: {verified_text}\n"
                f"Passed: {data['passed']}/{data['total']} ({data['pass_rate'] * 100:.0f}%)\n"
            )

            if data.get("badge"):
                response_text += f"Badge: {data['badge']}\n"

            if data.get("entity_id"):
                response_text += f"Entity: {data['entity_id']}\n"

            response_text += "\nChallenge Results:\n"
            for r in data["results"]:
                status = "PASS" if r["passed"] else "FAIL"
                response_text += f"  - {r['challenge_type']}: {status} ({r['response_time_ms']}ms/{r['time_limit_ms']}ms)\n"

            _get_session_token(session_id, consume=True)
            return [TextContent(type="text", text=response_text)]
        except httpx.HTTPStatusError as e:
            return [
                TextContent(
                    type="text", text=f"Error getting result: {e.response.text}"
                )
            ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "mettle_list_suites":
        try:
            suites = await api_call("/mettle/suites", auth=True)
            lines = ["METTLE v2 verification suites:\n"]
            for s in suites:
                flags = []
                if s.get("is_multi_round"):
                    flags.append("multi-round")
                if not s.get("available", True):
                    flags.append("unavailable")
                suffix = f"  [{', '.join(flags)}]" if flags else ""
                lines.append(
                    f"  {s.get('suite_number')}. {s.get('name')} ({s.get('display_name')}){suffix}"
                )
                lines.append(f"       {s.get('description', '')}")
            lines.append("\nStart one with mettle_start_v2_session (suites=[...]).")
            return [TextContent(type="text", text="\n".join(lines))]
        except httpx.HTTPStatusError as e:
            return [
                TextContent(
                    type="text", text=f"Error listing suites: {e.response.text}"
                )
            ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "mettle_start_v2_session":
        try:
            payload: dict[str, Any] = {
                "suites": arguments.get("suites", ["all"]),
                "difficulty": arguments.get("difficulty", "standard"),
            }
            if arguments.get("entity_id"):
                payload["entity_id"] = arguments["entity_id"]
            if arguments.get("vcp_token"):
                payload["vcp_token"] = arguments["vcp_token"]
            if arguments.get("allow_third_party_llm") is True:
                payload["allow_third_party_llm"] = True

            data = await api_call("/mettle/sessions", "POST", payload, auth=True)
            challenges = data.get("challenges", {})
            lines = [
                "METTLE v2 session started.\n",
                f"Session ID: {data.get('session_id')}",
                f"Suites: {', '.join(data.get('suites', []))}",
                f"Time budget: {data.get('time_budget_ms')}ms",
                f"Expires: {data.get('expires_at')}",
                "\nChallenges (answer each suite with mettle_verify_suite):",
            ]
            for suite_name, cdata in challenges.items():
                lines.append(f"  {suite_name}:")
                lines.append(f"    {json.dumps(cdata)}")
            return [TextContent(type="text", text="\n".join(lines))]
        except httpx.HTTPStatusError as e:
            return [
                TextContent(
                    type="text", text=f"Error starting v2 session: {e.response.text}"
                )
            ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "mettle_verify_suite":
        try:
            session_id = _safe_id(arguments["session_id"], "session_id")
            suite = _safe_id(arguments["suite"], "suite")
            data = await api_call(
                f"/mettle/sessions/{session_id}/verify",
                "POST",
                {"suite": suite, "answers": arguments["answers"]},
                auth=True,
            )
            passed = "PASSED" if data.get("passed") else "FAILED"
            lines = [
                f"Suite '{data.get('suite')}': {passed}",
                f"Score: {data.get('score')}",
            ]
            if data.get("details"):
                lines.append(f"Details: {json.dumps(data['details'])}")
            lines.append(
                "\nGet the final tier + attestation with mettle_get_v2_result."
            )
            return [TextContent(type="text", text="\n".join(lines))]
        except httpx.HTTPStatusError as e:
            return [
                TextContent(
                    type="text", text=f"Error verifying suite: {e.response.text}"
                )
            ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "mettle_get_v2_result":
        try:
            session_id = _safe_id(arguments["session_id"], "session_id")
            include_vcp = arguments.get("include_vcp", True)
            data = await api_call(
                f"/mettle/sessions/{session_id}/result",
                params={"include_vcp": str(include_vcp).lower()},
                auth=True,
            )
            passed = "PASSED" if data.get("overall_passed") else "NOT PASSED"
            lines = [
                "METTLE v2 Result",
                "=" * 30,
                f"Status: {data.get('status')}",
                f"Overall: {passed}",
                f"Tier: {data.get('tier') or 'none'}",
                f"Suites completed: {', '.join(data.get('suites_completed', []))}",
            ]
            attestation = data.get("vcp_attestation")
            if attestation:
                lines.append("\nVCP attestation (presentable credential):")
                lines.append(json.dumps(attestation, indent=2))
            gov = data.get("governance_attestation")
            if gov:
                lines.append(
                    f"\nGovernance attestation: framework={gov.get('framework')}"
                )
            return [TextContent(type="text", text="\n".join(lines))]
        except httpx.HTTPStatusError as e:
            return [
                TextContent(
                    type="text", text=f"Error getting v2 result: {e.response.text}"
                )
            ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _handle_list_tools(
    _context: ServerRequestContext[Any], _params: PaginatedRequestParams | None
) -> ListToolsResult:
    """Adapt METTLE's transport-independent listing to MCP 2's low-level API."""
    return ListToolsResult(tools=await list_tools())


async def _handle_call_tool(
    _context: ServerRequestContext[Any], params: CallToolRequestParams
) -> CallToolResult:
    """Adapt an MCP 2 request model to the transport-independent tool handler."""
    content: list[ContentBlock] = list(
        await call_tool(params.name, params.arguments or {})
    )
    return CallToolResult(content=content)


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
