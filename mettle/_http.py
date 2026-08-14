"""Streamable HTTP transport for the METTLE MCP server — additive.

stdio remains the default transport (``mettle-mcp``); this module is only
reached via ``mettle-mcp --transport http``. The MCP tools themselves are
transport-agnostic and are not touched here — this is purely an alternate run
harness for hosted deployments (Render, Smithery, Docker MCP gateway).

Hardening (see ``build_http_app``):

* Stateless session manager — a fresh transport per request, no persisted
  sessions, so there is no idle-session table to exhaust.
* POST-only — GET/DELETE get an instant 405 rather than holding an idle SSE
  stream open (connection-exhaustion DoS).
* Body size cap — oversize requests are rejected before being buffered.
* Clean JSON-RPC errors — malformed bodies yield ``-32700`` with no stack
  trace, install path, or dependency version leaking to the client (CWE-209).
* Bind guard — refuses to bind a non-loopback interface unless
  ``METTLE_MCP_ALLOW_INSECURE_HTTP=true`` is set, because the transport carries
  no authentication of its own.

Extra deps (declared in the ``mcp`` optional-dependencies group): ``uvicorn``,
``starlette``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ipaddress
import json
import os
import re
import time
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from urllib.parse import urlsplit

import httpx
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import Tool
from mettle import __version__
from mettle.mcp_context import caller_api_key, caller_principal, http_request_active
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

# JSON-RPC 2.0 error codes (mcp.types mirrors these; inlined to keep this
# module importable without pulling the type module in for two constants).
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600

#: Largest request body we will accept, in bytes. MCP tool calls here carry
#: challenge answers and small JSON objects; 1 MiB is generous. A cap matters
#: because the session manager buffers the body before dispatch.
MAX_BODY_BYTES = 1_048_576

#: Set to "true" to permit binding a non-loopback interface. Only appropriate
#: when something else fronts the server and handles client authentication
#: (Smithery's gateway, a reverse proxy). Never set it on a public bind.
ALLOW_INSECURE_ENV = "METTLE_MCP_ALLOW_INSECURE_HTTP"
ALLOWED_HOSTS_ENV = "METTLE_MCP_ALLOWED_HOSTS"
ALLOWED_ORIGINS_ENV = "METTLE_MCP_ALLOWED_ORIGINS"
REQUESTS_PER_MINUTE_ENV = "METTLE_MCP_REQUESTS_PER_MINUTE"
GLOBAL_REQUESTS_PER_MINUTE_ENV = "METTLE_MCP_MAX_GLOBAL_REQUESTS_PER_MINUTE"
MAX_PRINCIPALS_ENV = "METTLE_MCP_MAX_PRINCIPALS"
MAX_CONCURRENT_ENV = "METTLE_MCP_MAX_CONCURRENT_PER_CALLER"
MAX_GLOBAL_CONCURRENT_ENV = "METTLE_MCP_MAX_GLOBAL_CONCURRENT"

_LOOPBACK_HOSTNAMES = frozenset({"localhost"})
_BEARER_RE = re.compile(r"[^\s\x00-\x1f\x7f]{16,512}")

ToolProvider = Callable[[], Awaitable[list[Tool]]]
BearerValidator = Callable[[str], Awaitable[bool]]


def _jsonrpc_error(code: int, message: str, status_code: int, **kwargs) -> JSONResponse:
    """Build a JSON-RPC error response carrying no internal detail.

    ``message`` is always a fixed string chosen by us, never an exception's
    ``str()`` — exception text can embed file paths and dependency versions
    (CWE-209: information exposure through an error message).
    """
    return JSONResponse(
        {"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": None},
        status_code=status_code,
        **kwargs,
    )


def is_loopback_host(host: str) -> bool:
    """True if ``host`` names the loopback interface only.

    Accepts the literal addresses (``127.0.0.1``, ``::1``, anything in
    ``127.0.0.0/8``) as well as the ``localhost`` hostname. A wildcard bind
    (``0.0.0.0``, ``::``) is explicitly NOT loopback — that is the case the
    guard exists to catch.
    """
    candidate = host.strip().lower().strip("[]")
    if candidate in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        # An unresolvable/other hostname is not provably loopback; treat it as
        # exposed and make the operator opt in.
        return False


def insecure_bind_allowed() -> bool:
    """True if the operator has opted into non-loopback binds."""
    return os.environ.get(ALLOW_INSECURE_ENV, "").strip().lower() == "true"


def _csv_environment(name: str) -> set[str]:
    return {
        item.strip().lower()
        for item in os.environ.get(name, "").split(",")
        if item.strip()
    }


def _positive_int_environment(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _request_hostname(host_header: str) -> str | None:
    """Return a normalized hostname only for an unambiguous Host value."""
    try:
        parsed = urlsplit(f"//{host_header}")
        if (
            not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
        ):
            return None
        return parsed.hostname.lower()
    except ValueError:
        return None


def _bearer_token(headers: Headers) -> str | None:
    configured_header = headers.get("x-mettle-api-key", "").strip()
    if configured_header:
        return configured_header if _BEARER_RE.fullmatch(configured_header) else None
    authorization = headers.get("authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token if _BEARER_RE.fullmatch(token) else None


def _host_allowed(hostname: str, allowed: set[str]) -> bool:
    """Match exact hosts and explicit DNS suffix wildcards only."""
    for candidate in allowed:
        if hostname == candidate:
            return True
        if candidate.startswith("*.") and hostname.endswith(candidate[1:]):
            return True
    return False


async def _validate_bearer_upstream(token: str) -> bool:
    """Validate an HTTP caller against the canonical METTLE API authority."""
    api_url = os.environ.get("METTLE_API_URL", "https://mettle.sh/api").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
            response = await client.get(
                f"{api_url}/mettle/suites",
                headers={"Authorization": f"Bearer {token}"},
            )
    except httpx.HTTPError as exc:
        raise RuntimeError("MCP authentication authority is unavailable") from exc
    if response.status_code == 200:
        return True
    if response.status_code in {401, 403}:
        return False
    raise RuntimeError("MCP authentication authority is unavailable")


def check_bind_allowed(host: str) -> None:
    """Reject ambiguous or exposed binds without an explicit host policy."""
    if not host.strip():
        raise RuntimeError("refusing an empty MCP bind host")
    if is_loopback_host(host):
        return
    if insecure_bind_allowed() and _csv_environment(ALLOWED_HOSTS_ENV):
        return
    raise RuntimeError(
        f"refusing to bind {host!r}: set {ALLOW_INSECURE_ENV}=true and configure "
        f"an explicit {ALLOWED_HOSTS_ENV} value for an authenticated public endpoint"
    )


async def _read_capped_body(request: Request) -> bytes | None:
    """Return the request body, or ``None`` if it exceeds :data:`MAX_BODY_BYTES`.

    Streams rather than calling ``request.body()`` so an oversize payload is
    abandoned mid-flight instead of being fully buffered into memory.
    """
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > MAX_BODY_BYTES:
                return None
        except ValueError:
            return None

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def build_server_card(tools: list[Tool]) -> dict[str, object]:
    """Return Smithery's static MCP server card from the canonical tool models.

    Deriving the card from the same :class:`mcp.types.Tool` objects returned by
    ``tools/list`` prevents a stale discovery surface from reintroducing a
    removed tool. The card contains public schemas only and no runtime
    configuration or credential values.
    """
    return {
        "serverInfo": {"name": "mettle", "version": __version__},
        "authentication": {
            "required": True,
            "schemes": [
                {
                    "type": "http",
                    "scheme": "bearer",
                    "description": "A caller-owned METTLE API key",
                }
            ],
        },
        "tools": [
            tool.model_dump(mode="json", by_alias=True, exclude_none=True)
            for tool in tools
        ],
        "resources": [],
        "prompts": [],
    }


def build_http_app(
    server: Server,
    tool_provider: ToolProvider,
    *,
    bearer_validator: BearerValidator | None = None,
    allowed_hosts: set[str] | None = None,
    allowed_origins: set[str] | None = None,
) -> Starlette:
    """Wrap a low-level MCP ``Server`` in a Starlette ASGI app over Streamable HTTP.

    Stateless mode (a fresh transport per request, no persisted sessions) avoids
    the idle-session / held-event-stream exhaustion class of DoS. ``debug=False``
    ensures Starlette never renders a stack trace to a client (CWE-209).
    """
    manager = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        json_response=True,
        stateless=True,
    )
    validate_bearer = bearer_validator or _validate_bearer_upstream
    accepted_hosts = {
        item.lower()
        for item in (
            allowed_hosts
            if allowed_hosts is not None
            else _LOOPBACK_HOSTNAMES | _csv_environment(ALLOWED_HOSTS_ENV)
        )
    }
    accepted_hosts.update({"127.0.0.1", "::1"})
    accepted_origins = {
        item.rstrip("/").lower()
        for item in (
            allowed_origins
            if allowed_origins is not None
            else _csv_environment(ALLOWED_ORIGINS_ENV)
        )
    }
    requests_per_minute = _positive_int_environment(REQUESTS_PER_MINUTE_ENV, 60)
    global_requests_per_minute = _positive_int_environment(
        GLOBAL_REQUESTS_PER_MINUTE_ENV, 600
    )
    maximum_principals = _positive_int_environment(MAX_PRINCIPALS_ENV, 600)
    maximum_concurrent = _positive_int_environment(MAX_CONCURRENT_ENV, 4)
    maximum_global_concurrent = _positive_int_environment(MAX_GLOBAL_CONCURRENT_ENV, 64)
    budget_lock = asyncio.Lock()
    request_times: dict[str, deque[float]] = {}
    global_request_times: deque[float] = deque()
    active_requests: dict[str, int] = {}
    global_active = 0

    async def handle_mcp(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal global_active
        # Stateless mode has no session, so GET (the server->client SSE stream)
        # and DELETE (teardown) are meaningless — and a GET holds an idle
        # event-stream open indefinitely (a connection-exhaustion DoS). Accept
        # only POST; reject the rest with a clean JSON-RPC 405. Defense in
        # depth: a front proxy should also restrict methods, but the app must
        # be safe when run alone.
        if scope.get("method") != "POST":
            response = _jsonrpc_error(
                _INVALID_REQUEST,
                "Only POST is supported",
                405,
                headers={"Allow": "POST"},
            )
            await response(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_host = _request_hostname(headers.get("host", ""))
        if request_host is None or not _host_allowed(request_host, accepted_hosts):
            response = _jsonrpc_error(_INVALID_REQUEST, "Host is not allowed", 421)
            await response(scope, receive, send)
            return

        origin = headers.get("origin")
        if origin and origin.rstrip("/").lower() not in accepted_origins:
            response = _jsonrpc_error(_INVALID_REQUEST, "Origin is not allowed", 403)
            await response(scope, receive, send)
            return

        media_type = headers.get("content-type", "").partition(";")[0].strip().lower()
        if media_type != "application/json":
            response = _jsonrpc_error(
                _INVALID_REQUEST, "Content-Type must be application/json", 415
            )
            await response(scope, receive, send)
            return

        token = _bearer_token(headers)
        if token is None:
            response = _jsonrpc_error(
                _INVALID_REQUEST,
                "Bearer authorization is required",
                401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return

        principal = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = time.monotonic()
        acquired = False
        async with budget_lock:
            global_active_count = global_active
            while global_request_times and now - global_request_times[0] >= 60:
                global_request_times.popleft()
            for known_principal, known_times in tuple(request_times.items()):
                while known_times and now - known_times[0] >= 60:
                    known_times.popleft()
                if not known_times and active_requests.get(known_principal, 0) == 0:
                    request_times.pop(known_principal, None)
                    active_requests.pop(known_principal, None)
            recent = request_times.get(principal)
            active = active_requests.get(principal, 0)
            if len(global_request_times) >= global_requests_per_minute:
                budget_error = "Service request budget exceeded"
            elif recent is None and len(request_times) >= maximum_principals:
                budget_error = "Service principal budget exceeded"
            else:
                recent = request_times.setdefault(principal, deque())
                budget_error = None
            if budget_error is None:
                assert recent is not None
                if len(recent) >= requests_per_minute:
                    budget_error = "Caller request budget exceeded"
                elif active >= maximum_concurrent:
                    budget_error = "Caller concurrency budget exceeded"
                elif global_active_count >= maximum_global_concurrent:
                    budget_error = "Service concurrency budget exceeded"
            if budget_error is None:
                assert recent is not None
                recent.append(now)
                global_request_times.append(now)
                active_requests[principal] = active + 1
                global_active += 1
                acquired = True
        if budget_error is not None:
            response = _jsonrpc_error(
                _INVALID_REQUEST,
                budget_error,
                429,
                headers={"Retry-After": "1"},
            )
            await response(scope, receive, send)
            return

        try:
            try:
                authenticated = await validate_bearer(token)
            except Exception:
                response = _jsonrpc_error(
                    _INVALID_REQUEST,
                    "Authentication authority is unavailable",
                    503,
                )
                await response(scope, receive, send)
                return
            if not authenticated:
                response = _jsonrpc_error(
                    _INVALID_REQUEST,
                    "Bearer authorization is invalid",
                    401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return

            # Buffer the body ourselves so we can enforce the size cap and return
            # a clean parse error, then replay it to the session manager.
            request = Request(scope, receive)
            body = await _read_capped_body(request)
            if body is None:
                response = _jsonrpc_error(
                    _INVALID_REQUEST,
                    f"Request body exceeds {MAX_BODY_BYTES} bytes",
                    413,
                )
                await response(scope, receive, send)
                return

            try:
                json.loads(body)
            except (ValueError, UnicodeDecodeError):
                response = _jsonrpc_error(_PARSE_ERROR, "Parse error", 400)
                await response(scope, receive, send)
                return

            key_token = caller_api_key.set(token)
            principal_token = caller_principal.set(principal)
            http_token = http_request_active.set(True)
            try:
                await manager.handle_request(scope, _replay(body), send)
            finally:
                http_request_active.reset(http_token)
                caller_principal.reset(principal_token)
                caller_api_key.reset(key_token)
        finally:
            if acquired:
                async with budget_lock:
                    remaining = active_requests.get(principal, 1) - 1
                    if remaining > 0:
                        active_requests[principal] = remaining
                    else:
                        active_requests.pop(principal, None)
                    global_active -= 1

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def server_card(_request: Request) -> JSONResponse:
        return JSONResponse(
            build_server_card(await tool_provider()),
            headers={"Cache-Control": "public, max-age=300"},
        )

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncGenerator[None]:
        # run() sets up the manager's task group; it must live for the app's
        # lifetime and may only be entered once per instance.
        async with manager.run():
            yield

    # Mount at /mcp. A bare `/mcp` request 307-redirects to `/mcp/`; this is the
    # reference MCP behaviour and SDK/httpx clients follow it transparently.
    return Starlette(
        debug=False,
        routes=[
            Route("/health", health, methods=["GET"]),
            Route(
                "/.well-known/mcp/server-card.json",
                server_card,
                methods=["GET"],
            ),
            Mount("/mcp", app=handle_mcp),
        ],
        lifespan=lifespan,
    )


def _replay(body: bytes) -> Receive:
    """Return an ASGI ``receive`` callable that serves an already-read body."""
    sent = False

    async def receive() -> dict:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


def resolve_port(explicit: int | None) -> int:
    """Resolve the listen port: explicit ``--port`` > ``$PORT`` > 8080.

    Hosted platforms (Render, Smithery) inject ``$PORT``, so the CLI flag must
    default to ``None`` rather than a number — otherwise the flag always wins
    and the injected port is silently ignored.
    """
    if explicit is not None:
        return explicit
    env_port = os.environ.get("PORT", "").strip()
    if env_port:
        try:
            return int(env_port)
        except ValueError as exc:
            raise RuntimeError(
                f"invalid PORT environment variable: {env_port!r}"
            ) from exc
    return 8080


def run_http(
    server: Server,
    host: str,
    port: int | None,
    tool_provider: ToolProvider,
) -> None:  # pragma: no cover
    """Serve ``server`` over Streamable HTTP (blocking).

    ``uvicorn`` is imported lazily so stdio-only runs never require it.
    """
    import uvicorn

    resolved_port = resolve_port(port)
    check_bind_allowed(host)
    if os.environ.get("METTLE_API_KEY") or os.environ.get("METTLE_API_KEYS"):
        raise RuntimeError(
            "MCP HTTP must use each caller's Bearer API key; remove the shared "
            "METTLE_API_KEY/METTLE_API_KEYS process credential"
        )
    uvicorn.run(
        build_http_app(server, tool_provider),
        host=host,
        port=resolved_port,
        log_level="info",
        access_log=False,
    )
