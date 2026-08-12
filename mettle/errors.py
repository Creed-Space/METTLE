"""Stable public error categories for HTTP boundaries."""

from __future__ import annotations

ERROR_CODES_BY_STATUS = {
    400: "invalid_request",
    401: "authentication_required",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "request_too_large",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    502: "upstream_error",
    503: "dependency_unavailable",
    504: "upstream_timeout",
}


def error_code_for_status(status_code: int) -> str:
    """Return a stable category without exposing an internal exception name."""
    return ERROR_CODES_BY_STATUS.get(status_code, "request_error")
