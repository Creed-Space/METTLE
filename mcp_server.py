#!/usr/bin/env python3
"""Bounded Model Context Protocol adapter for the METTLE legacy API."""

import asyncio
import json as jsonlib
import math
import os
import re
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

API_URL = os.getenv("METTLE_API_URL", "https://mettle.sh/api")
MAX_UPSTREAM_RESPONSE_BYTES = 256 * 1024
_STREAM_CHUNK_BYTES = 16 * 1024
_SESSION_ID_PATTERN = r"^ses_[a-f0-9]{24}$"
_CHALLENGE_ID_PATTERN = r"^mtl_[a-f0-9]{24}$"
_BEARER_VALUE_PATTERN = r"^[A-Za-z0-9_-]{43}$"
_ENTITY_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"


def _string_schema(
    *,
    minimum: int = 0,
    maximum: int,
    pattern: str | None = None,
    description: str,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "string",
        "minLength": minimum,
        "maxLength": maximum,
        "description": description,
    }
    if pattern is not None:
        schema["pattern"] = pattern
    return schema


TOOL_INPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "mettle_start_session": {
        "type": "object",
        "properties": {
            "difficulty": {
                "type": "string",
                "enum": ["basic", "full"],
                "minLength": 4,
                "maxLength": 5,
                "description": "Verification difficulty level",
                "default": "basic",
            },
            "entity_id": _string_schema(
                minimum=1,
                maximum=128,
                pattern=_ENTITY_ID_PATTERN,
                description="Optional identifier for this AI agent",
            ),
        },
        "additionalProperties": False,
    },
    "mettle_answer_challenge": {
        "type": "object",
        "properties": {
            "session_id": _string_schema(
                minimum=28,
                maximum=28,
                pattern=_SESSION_ID_PATTERN,
                description="Session ID from mettle_start_session",
            ),
            "session_token": _string_schema(
                minimum=43,
                maximum=43,
                pattern=_BEARER_VALUE_PATTERN,
                description="Bearer token from mettle_start_session",
            ),
            "challenge_id": _string_schema(
                minimum=28,
                maximum=28,
                pattern=_CHALLENGE_ID_PATTERN,
                description="Challenge ID to answer",
            ),
            "answer": _string_schema(
                maximum=1024,
                description="Your answer to the challenge",
            ),
        },
        "required": ["session_id", "session_token", "challenge_id", "answer"],
        "additionalProperties": False,
    },
    "mettle_get_result": {
        "type": "object",
        "properties": {
            "session_id": _string_schema(
                minimum=28,
                maximum=28,
                pattern=_SESSION_ID_PATTERN,
                description="Session ID to get results for",
            ),
            "session_token": _string_schema(
                minimum=43,
                maximum=43,
                pattern=_BEARER_VALUE_PATTERN,
                description="Bearer token from mettle_start_session",
            ),
        },
        "required": ["session_id", "session_token"],
        "additionalProperties": False,
    },
}

server = Server("mettle")
http_client = httpx.AsyncClient(timeout=30.0)


class UpstreamResponseError(RuntimeError):
    """The configured API returned an unusable bounded response."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


async def api_call(
    endpoint: str,
    method: str = "GET",
    json: dict[str, Any] | None = None,
    session_token: str | None = None,
) -> dict[str, Any]:
    """Stream one METTLE API response through a strict byte and JSON boundary."""
    url = f"{API_URL}{endpoint}"
    headers = {"X-Session-Token": session_token} if session_token else None
    request_kwargs: dict[str, Any] = {"headers": headers}
    if method != "GET":
        request_kwargs["json"] = json

    async with http_client.stream(method, url, **request_kwargs) as response:
        response.raise_for_status()
        raw = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=_STREAM_CHUNK_BYTES):
            if len(raw) + len(chunk) > MAX_UPSTREAM_RESPONSE_BYTES:
                raise UpstreamResponseError("METTLE API response exceeded its limit")
            raw.extend(chunk)

    try:
        data = jsonlib.loads(
            bytes(raw),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeError, ValueError, TypeError) as exc:
        raise UpstreamResponseError("METTLE API returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise UpstreamResponseError("METTLE API returned a non-object response")
    return data


def _validate_arguments(name: str, arguments: object) -> str | None:
    """Validate direct handler calls against the advertised JSON schemas."""
    schema = TOOL_INPUT_SCHEMAS.get(name)
    if schema is None:
        return "unknown tool"
    if not isinstance(arguments, dict):
        return "arguments must be an object"

    properties: dict[str, dict[str, Any]] = schema["properties"]
    unknown = set(arguments) - set(properties)
    if unknown:
        return "unsupported argument"
    for required in schema.get("required", []):
        if required not in arguments:
            return f"missing required argument: {required}"

    for key, value in arguments.items():
        field = properties[key]
        if field.get("type") == "string":
            if not isinstance(value, str):
                return f"{key} must be a string"
            if len(value) < field.get("minLength", 0):
                return f"{key} is too short"
            if len(value) > field["maxLength"]:
                return f"{key} is too long"
            if "enum" in field and value not in field["enum"]:
                return f"{key} is not an allowed value"
            if pattern := field.get("pattern"):
                if re.fullmatch(pattern, value) is None:
                    return f"{key} has an invalid format"
    return None


def _result(text: str, *, is_error: bool = False) -> CallToolResult:
    """Create an explicit MCP tool result with the correct error signal."""
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        isError=is_error,
    )


def _response_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _response_string(
    value: object,
    label: str,
    *,
    maximum: int,
    pattern: str | None = None,
    minimum: int = 1,
) -> str:
    if (
        not isinstance(value, str)
        or not minimum <= len(value) <= maximum
        or (pattern is not None and re.fullmatch(pattern, value) is None)
    ):
        raise ValueError(f"{label} must be a bounded string")
    return value


def _response_integer(
    value: object, label: str, *, minimum: int = 0, maximum: int
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be a bounded integer")
    return value


def _response_boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _challenge_response(value: object) -> dict[str, Any]:
    challenge = _response_mapping(value, "challenge")
    return {
        "id": _response_string(
            challenge.get("id"),
            "challenge id",
            maximum=28,
            pattern=_CHALLENGE_ID_PATTERN,
        ),
        "type": _response_string(challenge.get("type"), "challenge type", maximum=64),
        "prompt": _response_string(
            challenge.get("prompt"), "challenge prompt", maximum=16_384
        ),
        "time_limit_ms": _response_integer(
            challenge.get("time_limit_ms"),
            "challenge time limit",
            minimum=1,
            maximum=86_400_000,
        ),
    }


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List the three bounded interactive verification tools."""
    descriptions = {
        "mettle_start_session": (
            "Start a METTLE verification session and receive its first challenge."
        ),
        "mettle_answer_challenge": (
            "Submit an answer and receive its result and any next challenge."
        ),
        "mettle_get_result": "Read the final result for a completed METTLE session.",
    }
    return [
        Tool(name=name, description=descriptions[name], inputSchema=schema)
        for name, schema in TOOL_INPUT_SCHEMAS.items()
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Validate, dispatch, and explicitly classify one MCP tool call."""
    validation_error = _validate_arguments(name, arguments)
    if validation_error is not None:
        return _result(f"Invalid METTLE tool call: {validation_error}", is_error=True)

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
            session_id = _response_string(
                data.get("session_id"),
                "session id",
                maximum=28,
                pattern=_SESSION_ID_PATTERN,
            )
            session_token = _response_string(
                data.get("session_token"),
                "session token",
                maximum=43,
                pattern=_BEARER_VALUE_PATTERN,
            )
            difficulty = _response_string(
                data.get("difficulty"), "difficulty", maximum=5, minimum=4
            )
            if difficulty not in {"basic", "full"}:
                raise ValueError("invalid difficulty")
            total_challenges = _response_integer(
                data.get("total_challenges"),
                "total challenges",
                minimum=1,
                maximum=100,
            )
            challenge = _challenge_response(data.get("current_challenge"))
            return _result(
                "METTLE session started!\n\n"
                f"Session ID: {session_id}\n"
                f"Session token: {session_token}\n"
                f"Difficulty: {difficulty}\n"
                f"Total challenges: {total_challenges}\n\n"
                "First Challenge:\n"
                f"  ID: {challenge['id']}\n"
                f"  Type: {challenge['type']}\n"
                f"  Prompt: {challenge['prompt']}\n"
                f"  Time limit: {challenge['time_limit_ms']}ms\n\n"
                "Use mettle_answer_challenge to submit your answer."
            )

        if name == "mettle_answer_challenge":
            data = await api_call(
                "/session/answer",
                "POST",
                {
                    "session_id": arguments["session_id"],
                    "challenge_id": arguments["challenge_id"],
                    "answer": arguments["answer"],
                },
                session_token=arguments["session_token"],
            )
            challenge_result = _response_mapping(data.get("result"), "result")
            challenge_passed = _response_boolean(
                challenge_result.get("passed"), "passed"
            )
            response_time_ms = _response_integer(
                challenge_result.get("response_time_ms"),
                "response time",
                maximum=86_400_000,
            )
            time_limit_ms = _response_integer(
                challenge_result.get("time_limit_ms"),
                "time limit",
                minimum=1,
                maximum=86_400_000,
            )
            session_complete = _response_boolean(
                data.get("session_complete"), "session complete"
            )
            passed_text = "PASSED" if challenge_passed else "FAILED"
            response_text = (
                f"Challenge Result: {passed_text}\n"
                f"Response time: {response_time_ms}ms "
                f"(limit: {time_limit_ms}ms)\n"
            )
            if session_complete:
                response_text += (
                    "\nSession complete! Challenges remaining: 0\n"
                    "Use mettle_get_result to see your final verification result."
                )
            else:
                challenges_remaining = _response_integer(
                    data.get("challenges_remaining"),
                    "challenges remaining",
                    minimum=1,
                    maximum=100,
                )
                next_challenge = _challenge_response(data.get("next_challenge"))
                response_text += (
                    f"\nChallenges remaining: {challenges_remaining}\n\n"
                    "Next Challenge:\n"
                    f"  ID: {next_challenge['id']}\n"
                    f"  Type: {next_challenge['type']}\n"
                    f"  Prompt: {next_challenge['prompt']}\n"
                    f"  Time limit: {next_challenge['time_limit_ms']}ms"
                )
            return _result(response_text)

        data = await api_call(
            f"/session/{arguments['session_id']}/result",
            session_token=arguments["session_token"],
        )
        verified = _response_boolean(data.get("verified"), "verified")
        passed_count = _response_integer(data.get("passed"), "passed", maximum=100)
        total = _response_integer(data.get("total"), "total", maximum=100)
        pass_rate_raw = data.get("pass_rate")
        if (
            isinstance(pass_rate_raw, bool)
            or not isinstance(pass_rate_raw, (int, float))
            or not math.isfinite(pass_rate_raw)
            or not 0.0 <= pass_rate_raw <= 1.0
            or passed_count > total
            or (total == 0 and (passed_count != 0 or pass_rate_raw != 0))
            or (
                total > 0
                and not math.isclose(pass_rate_raw, passed_count / total, abs_tol=1e-9)
            )
        ):
            raise ValueError("incoherent result totals")
        tier = _response_string(data.get("tier", "none"), "tier", maximum=16)
        if tier not in {"none", "bronze", "silver"}:
            raise ValueError("invalid result tier")
        raw_results = data.get("results")
        if not isinstance(raw_results, list) or len(raw_results) != total:
            raise ValueError("invalid result list")
        rendered_results: list[tuple[str, bool, int, int]] = []
        for raw_result in raw_results:
            challenge_result = _response_mapping(raw_result, "challenge result")
            rendered_results.append(
                (
                    _response_string(
                        challenge_result.get("challenge_type"),
                        "challenge type",
                        maximum=64,
                    ),
                    _response_boolean(challenge_result.get("passed"), "passed"),
                    _response_integer(
                        challenge_result.get("response_time_ms"),
                        "response time",
                        maximum=86_400_000,
                    ),
                    _response_integer(
                        challenge_result.get("time_limit_ms"),
                        "time limit",
                        minimum=1,
                        maximum=86_400_000,
                    ),
                )
            )
        entity_id = data.get("entity_id")
        if entity_id is not None:
            entity_id = _response_string(entity_id, "entity id", maximum=128)
        badge = data.get("badge")
        if badge is not None:
            badge = _response_string(badge, "badge", maximum=65_536)
        status_text = "VERIFIED" if verified else "NOT VERIFIED"
        response_text = (
            "METTLE Verification Result\n"
            f"{'=' * 30}\n\n"
            f"Status: {status_text}\n"
            f"Tier: {tier}\n"
            f"Passed: {passed_count}/{total} "
            f"({pass_rate_raw * 100:.0f}%)\n"
        )
        if entity_id is not None:
            response_text += f"Entity: {entity_id}\n"
        response_text += "\nChallenge Results:\n"
        for (
            challenge_type,
            result_passed,
            response_time,
            time_limit,
        ) in rendered_results:
            status = "PASS" if result_passed else "FAIL"
            response_text += (
                f"  - {challenge_type}: {status} ({response_time}ms/{time_limit}ms)\n"
            )
        if badge is not None:
            response_text += f"\nSigned credential issued:\n{badge}\n"
        return _result(response_text)
    except httpx.HTTPStatusError:
        return _result("METTLE API rejected the request.", is_error=True)
    except (httpx.HTTPError, UpstreamResponseError):
        return _result("METTLE API is temporarily unavailable.", is_error=True)
    except (KeyError, TypeError, ValueError):
        return _result("METTLE API returned an invalid response.", is_error=True)
    except Exception:
        return _result("METTLE tool execution failed.", is_error=True)


async def main():  # pragma: no cover
    """Run the MCP server and always release its module-owned HTTP client."""
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        await http_client.aclose()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
