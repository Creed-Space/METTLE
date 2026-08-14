"""Request-local identity for MCP transports.

The stdio transport uses its operator-configured API key. HTTP requests set
these context variables only after authoritative upstream authentication, so
tool calls cannot fall back to one process-wide credential or share session
bearer state across callers.
"""

from __future__ import annotations

from contextvars import ContextVar


caller_api_key: ContextVar[str | None] = ContextVar(
    "mettle_mcp_caller_api_key", default=None
)
caller_principal: ContextVar[str] = ContextVar(
    "mettle_mcp_caller_principal", default="stdio"
)
http_request_active: ContextVar[bool] = ContextVar(
    "mettle_mcp_http_request_active", default=False
)
