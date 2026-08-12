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

import contextlib
import ipaddress
import json
import os
from collections.abc import AsyncGenerator, Awaitable, Callable

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import Tool
from mettle import __version__
from starlette.applications import Starlette
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

_LOOPBACK_HOSTNAMES = frozenset({"localhost", ""})

ToolProvider = Callable[[], Awaitable[list[Tool]]]


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


def check_bind_allowed(host: str) -> None:
    """Raise ``RuntimeError`` if binding ``host`` would expose an unauthenticated server."""
    if is_loopback_host(host) or insecure_bind_allowed():
        return
    raise RuntimeError(
        f"refusing to bind {host!r}: the METTLE MCP HTTP transport has no "
        f"authentication of its own. Bind a loopback address behind a proxy, or "
        f"set {ALLOW_INSECURE_ENV}=true if a gateway in front of it authenticates clients."
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
        "authentication": {"required": False, "schemes": []},
        "tools": [
            tool.model_dump(mode="json", by_alias=True, exclude_none=True)
            for tool in tools
        ],
        "resources": [],
        "prompts": [],
    }


def build_http_app(server: Server, tool_provider: ToolProvider) -> Starlette:
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

    async def handle_mcp(scope: Scope, receive: Receive, send: Send) -> None:
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

        # Buffer the body ourselves so we can enforce the size cap and return a
        # clean parse error, then replay it to the session manager. Bodies that
        # reach here are already bounded by MAX_BODY_BYTES.
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
            # Fixed message: never echo the parser's exception, which can carry
            # payload fragments back to an unauthenticated caller.
            response = _jsonrpc_error(_PARSE_ERROR, "Parse error", 400)
            await response(scope, receive, send)
            return

        await manager.handle_request(scope, _replay(body), send)

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
    uvicorn.run(
        build_http_app(server, tool_provider),
        host=host,
        port=resolved_port,
        log_level="info",
        access_log=False,
    )
