"""Versioned, bounded control results for the METTLE MCP adapter.

The REST APIs remain the application authority. This module gives MCP callers a
stable machine-readable control vocabulary while the compatibility text remains
available to older hosts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

import httpx
from mcp.types import TextContent

SCHEMA_VERSION = "mettle-control-v1"
RECEIPT_VERSION = "mettle-control-receipt-v1"

RETRY_VALUES = [
    "safe_same_operation",
    "refresh_then_retry",
    "retry_after",
    "start_new_resource",
    "request_new_challenge",
    "do_not_retry",
    "operator_action_required",
]

ACTION_SCHEMA: dict[str, Any] = {
    "$id": "https://mettle.sh/schemas/control/v1/action.json",
    "type": "object",
    "additionalProperties": False,
    "required": ["operation", "title", "mutation", "idempotent"],
    "properties": {
        "operation": {"type": "string", "minLength": 1, "maxLength": 64},
        "title": {"type": "string", "minLength": 1, "maxLength": 160},
        "mutation": {"type": "boolean"},
        "idempotent": {"type": "boolean"},
        "arguments": {"type": "object"},
    },
}

SNAPSHOT_SCHEMA: dict[str, Any] = {
    "$id": "https://mettle.sh/schemas/control/v1/session-snapshot.json",
    "type": "object",
    "additionalProperties": True,
    "required": ["session_id", "profile", "status", "terminal"],
    "properties": {
        "session_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "profile": {"enum": ["quick", "authenticated"]},
        "status": {
            "enum": [
                "created",
                "ready",
                "in_progress",
                "completed",
                "expired",
                "cancelled",
            ]
        },
        "terminal": {"type": "boolean"},
        "current_round": {"type": ["integer", "null"], "minimum": 0},
        "suites": {"type": "array", "items": {"type": "string"}},
        "suites_completed": {"type": "array", "items": {"type": "string"}},
        "current_challenge": {"type": ["object", "null"]},
    },
}

RECEIPT_SCHEMA: dict[str, Any] = {
    "$id": "https://mettle.sh/schemas/control/v1/receipt.json",
    "type": "object",
    "additionalProperties": True,
    "required": ["receipt_version", "operation", "accepted"],
    "properties": {
        "receipt_version": {"const": RECEIPT_VERSION},
        "operation": {"type": "string", "minLength": 1, "maxLength": 64},
        "accepted": {"type": "boolean"},
    },
}

ERROR_SCHEMA: dict[str, Any] = {
    "$id": "https://mettle.sh/schemas/control/v1/error.json",
    "type": "object",
    "additionalProperties": False,
    "required": ["code", "message", "retry"],
    "properties": {
        "code": {"type": "string", "pattern": "^[a-z][a-z0-9_]{1,63}$"},
        "message": {"type": "string", "minLength": 1, "maxLength": 240},
        "retry": {"enum": RETRY_VALUES},
        "http_status": {"type": "integer", "minimum": 400, "maximum": 599},
        "retry_after_ms": {"type": "integer", "minimum": 0},
    },
}

CONTROL_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://mettle.sh/schemas/control/v1/envelope.json",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "operation",
        "outcome",
        "server_time",
        "data",
        "actions",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "operation": {"type": "string", "minLength": 1, "maxLength": 64},
        "outcome": {"enum": ["succeeded", "rejected", "failed"]},
        "server_time": {"type": "string", "format": "date-time"},
        "data": {"type": ["object", "array", "null"]},
        "snapshot": SNAPSHOT_SCHEMA,
        "actions": {"type": "array", "items": ACTION_SCHEMA},
        "receipt": RECEIPT_SCHEMA,
        "error": ERROR_SCHEMA,
    },
}


class ToolResponse(list[TextContent]):
    """Compatibility text plus MCP 2 structured content and error state."""

    def __init__(
        self,
        text: str,
        structured_content: dict[str, Any],
        *,
        is_error: bool = False,
    ) -> None:
        super().__init__([TextContent(type="text", text=text)])
        self.structured_content = structured_content
        self.is_error = is_error


def _server_time() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def receipt(operation: str, **facts: Any) -> dict[str, Any]:
    """Build a bounded mutation receipt without participant content."""
    return {
        "receipt_version": RECEIPT_VERSION,
        "operation": operation,
        "accepted": True,
        **facts,
    }


def action(
    operation: str,
    title: str,
    *,
    mutation: bool,
    idempotent: bool,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "operation": operation,
        "title": title,
        "mutation": mutation,
        "idempotent": idempotent,
    }
    if arguments:
        value["arguments"] = arguments
    return value


def success(
    operation: str,
    text: str,
    *,
    data: dict[str, Any] | list[Any] | None,
    snapshot: dict[str, Any] | None = None,
    actions: Iterable[dict[str, Any]] = (),
    mutation_receipt: dict[str, Any] | None = None,
) -> ToolResponse:
    envelope: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "operation": operation,
        "outcome": "succeeded",
        "server_time": _server_time(),
        "data": data,
        "actions": list(actions),
    }
    if snapshot is not None:
        envelope["snapshot"] = snapshot
    if mutation_receipt is not None:
        envelope["receipt"] = mutation_receipt
    return ToolResponse(text, envelope)


_HTTP_ERRORS: dict[int, tuple[str, str, str]] = {
    400: (
        "invalid_request",
        "The request was rejected by the session authority.",
        "refresh_then_retry",
    ),
    401: (
        "authentication_required",
        "Authentication is required or no longer valid.",
        "operator_action_required",
    ),
    403: (
        "forbidden",
        "The caller is not permitted to perform this operation.",
        "do_not_retry",
    ),
    404: (
        "resource_not_found",
        "The requested resource was not found or has expired.",
        "start_new_resource",
    ),
    409: (
        "state_conflict",
        "The resource state conflicts with this operation.",
        "refresh_then_retry",
    ),
    422: (
        "invalid_request",
        "The request does not match the operation schema.",
        "refresh_then_retry",
    ),
    429: ("rate_limited", "The service rate limit was reached.", "retry_after"),
    500: (
        "upstream_failure",
        "The session authority could not complete the operation.",
        "operator_action_required",
    ),
    502: (
        "upstream_unavailable",
        "The session authority is temporarily unavailable.",
        "retry_after",
    ),
    503: (
        "dependency_unavailable",
        "A required service is temporarily unavailable.",
        "retry_after",
    ),
    504: (
        "upstream_timeout",
        "The session authority did not respond in time.",
        "retry_after",
    ),
}

_SESSION_MUTATIONS = {
    "mettle_answer_challenge",
    "mettle_cancel_session",
    "mettle_submit_round",
    "mettle_verify_suite",
}
_RESOURCE_CREATION_OPERATIONS = {
    "mettle_start_session",
    "mettle_start_v2_session",
}


def _ambiguous_mutation_retry(operation: str, retry: str) -> str:
    """Prevent unsafe blind retry when a mutation may already have committed."""
    if operation in _RESOURCE_CREATION_OPERATIONS:
        return "operator_action_required"
    if operation in _SESSION_MUTATIONS:
        return "refresh_then_retry"
    return retry


def failure(operation: str, exc: BaseException) -> ToolResponse:
    """Map an exception to a fixed, non-reflective MCP control error."""
    http_status: int | None = None
    retry_after_ms: int | None = None
    outcome = "failed"
    if isinstance(exc, httpx.HTTPStatusError):
        http_status = exc.response.status_code
        code, message, retry = _HTTP_ERRORS.get(
            http_status,
            (
                "upstream_failure",
                "The session authority could not complete the operation.",
                "operator_action_required",
            ),
        )
        outcome = "rejected" if 400 <= http_status < 500 else "failed"
        if http_status >= 500:
            retry = _ambiguous_mutation_retry(operation, retry)
        raw_retry = exc.response.headers.get("Retry-After")
        if retry == "retry_after" and raw_retry and raw_retry.isdigit():
            retry_after_ms = min(int(raw_retry) * 1000, 86_400_000)
    elif isinstance(exc, ValueError):
        if str(exc).startswith("Unknown or expired session"):
            code = "capability_unavailable"
            message = "The hidden session capability is unavailable or expired."
            retry = "start_new_resource"
        else:
            code = "invalid_request"
            message = "The request or local session handle is invalid."
            retry = "refresh_then_retry"
        outcome = "rejected"
    elif isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        code = "upstream_timeout"
        message = "The session authority did not respond in time."
        retry = _ambiguous_mutation_retry(operation, "safe_same_operation")
    else:
        code = "internal_error"
        message = "The operation could not be completed."
        retry = "operator_action_required"

    error: dict[str, Any] = {"code": code, "message": message, "retry": retry}
    if http_status is not None:
        error["http_status"] = http_status
    if retry_after_ms is not None:
        error["retry_after_ms"] = retry_after_ms
    envelope: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "operation": operation,
        "outcome": outcome,
        "server_time": _server_time(),
        "data": None,
        "actions": [],
        "error": error,
    }
    return ToolResponse(
        f"METTLE control error [{code}]: {message}", envelope, is_error=True
    )


def normalize_status(value: object) -> str:
    """Map application states to the control-v1 state vocabulary."""
    status = str(value or "in_progress")
    if status in {"challenges_generated", "created"}:
        return "ready"
    if status in {"complete", "completed"}:
        return "completed"
    if status in {"expired", "cancelled", "in_progress"}:
        return status
    return "in_progress"


def session_actions(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive only actions that are valid from the caller-visible snapshot."""
    session_id = snapshot["session_id"]
    if snapshot["terminal"]:
        if snapshot["status"] == "completed":
            operation = (
                "mettle_get_result"
                if snapshot["profile"] == "quick"
                else "mettle_get_v2_result"
            )
            return [
                action(
                    operation,
                    "Read the completed session result",
                    mutation=False,
                    idempotent=True,
                    arguments={"session_id": session_id},
                )
            ]
        return []

    if snapshot["profile"] == "quick":
        challenge = snapshot.get("current_challenge")
        if isinstance(challenge, dict) and challenge.get("id"):
            return [
                action(
                    "mettle_answer_challenge",
                    "Answer the current quick challenge",
                    mutation=True,
                    idempotent=False,
                    arguments={
                        "session_id": session_id,
                        "challenge_id": challenge["id"],
                    },
                )
            ]
        return [
            action(
                "mettle_get_session",
                "Refresh the quick session",
                mutation=False,
                idempotent=True,
                arguments={"session_id": session_id, "profile": "quick"},
            )
        ]

    actions: list[dict[str, Any]] = [
        action(
            "mettle_get_session",
            "Refresh the authenticated session",
            mutation=False,
            idempotent=True,
            arguments={"session_id": session_id, "profile": "authenticated"},
        ),
        action(
            "mettle_cancel_session",
            "Cancel the authenticated session",
            mutation=True,
            idempotent=False,
            arguments={"session_id": session_id},
        ),
    ]
    current_round = snapshot.get("current_round")
    suites = snapshot.get("suites", [])
    completed = set(snapshot.get("suites_completed", []))
    for suite in reversed(suites):
        if suite in completed or suite == "novel-reasoning":
            continue
        actions.insert(
            0,
            action(
                "mettle_verify_suite",
                f"Submit the {suite} suite",
                mutation=True,
                idempotent=False,
                arguments={"session_id": session_id, "suite": suite},
            ),
        )
    if "novel-reasoning" in suites and "novel-reasoning" not in completed:
        actions.insert(
            0,
            action(
                "mettle_submit_round",
                "Submit the next novel-reasoning round",
                mutation=True,
                idempotent=False,
                arguments={
                    "session_id": session_id,
                    "round_num": int(current_round or 0) + 1,
                },
            ),
        )
    return actions
