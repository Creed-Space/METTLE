"""Tests for mettle._http — the Streamable HTTP transport for the MCP server.

Covers:
- A real over-the-wire handshake: bind a live uvicorn server, run the MCP
  initialize + tools/list round-trip, and assert the same seven tools the stdio
  arm advertises.
- The hardening branches, which are the reason this module exists: POST-only
  (405), malformed body (clean -32700, no stack trace), oversize body (413),
  and the non-loopback bind guard.
- The $PORT-vs---port precedence that hosted platforms depend on.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket

import httpx
import httpx2
import pytest

# The MCP server is an optional extra (`pip install 'mettle-verifier[mcp]'`),
# and the HTTP transport additionally needs starlette/uvicorn. Guard the whole
# module the same way tests/test_mcp_server.py does. CI installs
# requirements-mcp.txt, so these do not silently skip there.
pytest.importorskip("mcp")
pytest.importorskip("starlette")
pytest.importorskip("uvicorn")

from mettle import _http, mcp_server  # noqa: E402
from tests.test_mcp_server import EXPECTED_TOOLS  # noqa: E402


TEST_BEARER = "mtl_" + "a" * 32


async def _accept_test_bearer(token: str) -> bool:
    return token == TEST_BEARER


def _free_port() -> int:
    """Reserve an ephemeral port and hand back the number."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.asynccontextmanager
async def _live_server():
    """Run the HTTP app on a real loopback port; yield its /mcp URL."""
    import uvicorn

    port = _free_port()
    config = uvicorn.Config(
        _http.build_http_app(
            mcp_server.server,
            mcp_server.list_tools,
            bearer_validator=_accept_test_bearer,
        ),
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(200):  # up to ~10s for the socket to come up
            if server.started:
                break
            await asyncio.sleep(0.05)
        assert server.started, "uvicorn did not start"
        yield f"http://127.0.0.1:{port}/mcp/"
    finally:
        server.should_exit = True
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=10)


def _asgi_client() -> httpx.AsyncClient:
    """Client wired straight to the ASGI app.

    Usable for every check that our gate answers *before* delegating to the
    session manager, so no lifespan startup is required.
    """
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=_http.build_http_app(
                mcp_server.server,
                mcp_server.list_tools,
                bearer_validator=_accept_test_bearer,
                allowed_hosts={"testserver"},
            )
        ),
        base_url="http://testserver",
        headers={"Authorization": f"Bearer {TEST_BEARER}"},
    )


# === real handshake ===


async def test_http_handshake_lists_tools():
    """initialize + tools/list over Streamable HTTP returns the full tool set."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with _live_server() as url:
        async with httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {TEST_BEARER}"}
        ) as authenticated_client:
            async with streamable_http_client(
                url, http_client=authenticated_client
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()

    names = {t.name for t in result.tools}
    assert names == EXPECTED_TOOLS
    assert len(names) == 7


async def test_health_endpoint():
    async with _asgi_client() as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_server_card_matches_canonical_tool_surface():
    """Static discovery must stay identical to tools/list and exclude the solver."""
    async with _asgi_client() as client:
        response = await client.get("/.well-known/mcp/server-card.json")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "public, max-age=300"
    card = response.json()
    assert card["serverInfo"] == {"name": "mettle", "version": "0.4.3"}
    assert card["authentication"]["required"] is True
    assert card["authentication"]["schemes"][0]["scheme"] == "bearer"
    assert card["resources"] == []
    assert card["prompts"] == []

    card_tools = card["tools"]
    canonical_tools = [
        tool.model_dump(mode="json", by_alias=True, exclude_none=True)
        for tool in await mcp_server.list_tools()
    ]
    assert card_tools == canonical_tools
    names = {tool["name"] for tool in card_tools}
    assert names == EXPECTED_TOOLS
    assert len(names) == 7
    assert "mettle_auto_verify" not in names


async def test_replayed_request_body_disconnects_after_one_delivery():
    """The MCP adapter must not replay one request body more than once."""
    receive = _http._replay(b'{"jsonrpc":"2.0"}')
    first = await receive()
    second = await receive()
    assert first == {
        "type": "http.request",
        "body": b'{"jsonrpc":"2.0"}',
        "more_body": False,
    }
    assert second == {"type": "http.disconnect"}


# === hardening ===


@pytest.mark.parametrize("method", ["GET", "DELETE"])
async def test_non_post_methods_rejected_immediately(method):
    """GET would otherwise hold an idle SSE stream open (connection exhaustion)."""
    async with _asgi_client() as client:
        response = await client.request(method, "/mcp/")

    assert response.status_code == 405
    assert response.headers["allow"] == "POST"
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["error"]["code"] == -32600


async def test_malformed_body_returns_clean_parse_error():
    """Malformed JSON yields -32700 and leaks no internals (CWE-209)."""
    async with _asgi_client() as client:
        response = await client.post(
            "/mcp/",
            content=b"{not json at all",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == -32700
    assert body["error"]["message"] == "Parse error"
    # No stack trace, install path, or payload echo.
    raw = response.text
    for leak in ("Traceback", "site-packages", "/Users/", "not json at all"):
        assert leak not in raw


async def test_oversize_body_rejected():
    """A body above the cap is refused rather than buffered."""
    payload = b'{"padding":"' + b"A" * (_http.MAX_BODY_BYTES + 1024) + b'"}'
    async with _asgi_client() as client:
        response = await client.post(
            "/mcp/", content=payload, headers={"content-type": "application/json"}
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == -32600


async def test_oversize_body_rejected_without_content_length():
    """The streaming path must also cap when content-length is absent/chunked."""

    async def chunks():
        chunk = b"A" * 65536
        for _ in range((_http.MAX_BODY_BYTES // len(chunk)) + 2):
            yield chunk

    async with _asgi_client() as client:
        response = await client.post(
            "/mcp/", content=chunks(), headers={"content-type": "application/json"}
        )

    assert response.status_code == 413


async def test_bad_content_length_header_rejected():
    async with _asgi_client() as client:
        response = await client.post(
            "/mcp/",
            content=b"{}",
            headers={
                "content-length": "not-a-number",
                "content-type": "application/json",
            },
        )
    assert response.status_code == 413


async def test_valid_json_reaches_the_session_manager():
    """The gate must replay the buffered body, not swallow it."""
    async with _live_server() as url:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url + "/",
                content=json.dumps(
                    {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
                ),
                headers={
                    "content-type": "application/json",
                    "accept": "application/json, text/event-stream",
                    "authorization": f"Bearer {TEST_BEARER}",
                },
            )

    # Reached the MCP layer: a protocol-level answer, not our 400/405/413 gate.
    assert response.status_code == 200
    assert "tools" in response.text


async def test_mcp_requires_authoritative_bearer_authentication():
    async with _asgi_client() as client:
        missing = await client.post(
            "/mcp/",
            content=b"{}",
            headers={"Authorization": "", "Content-Type": "application/json"},
        )
        invalid = await client.post(
            "/mcp/",
            content=b"{}",
            headers={
                "Authorization": "Bearer mtl_" + "b" * 32,
                "Content-Type": "application/json",
            },
        )

    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert invalid.status_code == 401


async def test_mcp_rejects_untrusted_host_origin_and_content_type():
    async with _asgi_client() as client:
        bad_host = await client.post(
            "/mcp/",
            content=b"{}",
            headers={"Host": "attacker.example", "Content-Type": "application/json"},
        )
        bad_origin = await client.post(
            "/mcp/",
            content=b"{}",
            headers={
                "Origin": "https://attacker.example",
                "Content-Type": "application/json",
            },
        )
        text_plain = await client.post(
            "/mcp/",
            content=b"{}",
            headers={"Content-Type": "text/plain"},
        )

    assert bad_host.status_code == 421
    assert bad_origin.status_code == 403
    assert text_plain.status_code == 415


async def test_mcp_enforces_per_caller_request_budget(monkeypatch):
    monkeypatch.setenv(_http.REQUESTS_PER_MINUTE_ENV, "1")
    app = _http.build_http_app(
        mcp_server.server,
        mcp_server.list_tools,
        bearer_validator=_accept_test_bearer,
        allowed_hosts={"testserver"},
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        headers={"Authorization": f"Bearer {TEST_BEARER}"},
    ) as client:
        first = await client.post(
            "/mcp/", content=b"{bad", headers={"Content-Type": "application/json"}
        )
        second = await client.post(
            "/mcp/", content=b"{bad", headers={"Content-Type": "application/json"}
        )

    assert first.status_code == 400
    assert second.status_code == 429


async def test_rotating_invalid_bearers_share_a_global_authentication_budget(
    monkeypatch,
):
    monkeypatch.setenv(_http.REQUESTS_PER_MINUTE_ENV, "100")
    monkeypatch.setenv(_http.GLOBAL_REQUESTS_PER_MINUTE_ENV, "3")
    monkeypatch.setenv(_http.MAX_PRINCIPALS_ENV, "3")
    validator_calls = 0

    async def reject_bearer(_token: str) -> bool:
        nonlocal validator_calls
        validator_calls += 1
        return False

    app = _http.build_http_app(
        mcp_server.server,
        mcp_server.list_tools,
        bearer_validator=reject_bearer,
        allowed_hosts={"testserver"},
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        responses = []
        for index in range(5):
            responses.append(
                await client.post(
                    "/mcp/",
                    content=b"{}",
                    headers={
                        "Authorization": "Bearer mtl_" + str(index) * 32,
                        "Content-Type": "application/json",
                    },
                )
            )

    assert [response.status_code for response in responses] == [401, 401, 401, 429, 429]
    assert validator_calls == 3


# === bind guard ===


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.5", "LOCALHOST"]
)
def test_loopback_hosts_allowed(host, monkeypatch):
    monkeypatch.delenv(_http.ALLOW_INSECURE_ENV, raising=False)
    assert _http.is_loopback_host(host) is True
    _http.check_bind_allowed(host)  # must not raise


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.0.0.4", "example.com"])  # noqa: S104
def test_non_loopback_bind_refused_without_optin(host, monkeypatch):
    """The security assertion: no unauthenticated exposed bind by default."""
    monkeypatch.delenv(_http.ALLOW_INSECURE_ENV, raising=False)
    assert _http.is_loopback_host(host) is False
    with pytest.raises(RuntimeError, match=_http.ALLOW_INSECURE_ENV):
        _http.check_bind_allowed(host)


def test_non_loopback_bind_allowed_with_optin(monkeypatch):
    monkeypatch.setenv(_http.ALLOW_INSECURE_ENV, "true")
    monkeypatch.setenv(_http.ALLOWED_HOSTS_ENV, "mcp.example")
    _http.check_bind_allowed("0.0.0.0")  # noqa: S104 — the opted-in gateway case


def test_empty_bind_host_is_never_loopback(monkeypatch):
    monkeypatch.setenv(_http.ALLOW_INSECURE_ENV, "true")
    monkeypatch.setenv(_http.ALLOWED_HOSTS_ENV, "mcp.example")
    assert _http.is_loopback_host("") is False
    with pytest.raises(RuntimeError, match="empty MCP bind host"):
        _http.check_bind_allowed("")


@pytest.mark.parametrize("value", ["invalid", "0", "-1"])
def test_request_budget_environment_requires_a_positive_integer(value, monkeypatch):
    """Invalid budget input must stop startup rather than disable the limiter."""
    monkeypatch.setenv(_http.REQUESTS_PER_MINUTE_ENV, value)
    with pytest.raises(RuntimeError, match="must be a positive integer"):
        _http._positive_int_environment(_http.REQUESTS_PER_MINUTE_ENV, 60)


@pytest.mark.parametrize("host", ["user@example.com", "[malformed"])
def test_request_hostname_rejects_ambiguous_authority_values(host):
    """Credential-bearing or unparsable Host values are never normalized as trusted."""
    assert _http._request_hostname(host) is None


def test_host_policy_accepts_only_explicit_suffix_wildcards():
    assert _http._host_allowed("mcp.example.com", {"*.example.com"}) is True
    assert _http._host_allowed("example.com", {"*.example.com"}) is False


@pytest.mark.parametrize("value", ["", "false", "1", "yes", "TRUE "])
def test_optin_requires_exactly_true(value, monkeypatch):
    """Only "true" (case/space-insensitive) opts in — not any truthy-looking string."""
    monkeypatch.setenv(_http.ALLOW_INSECURE_ENV, value)
    expected = value.strip().lower() == "true"
    assert _http.insecure_bind_allowed() is expected


# === port resolution ===


def test_explicit_port_wins(monkeypatch):
    monkeypatch.setenv("PORT", "9999")
    assert _http.resolve_port(1234) == 1234


def test_port_env_used_when_flag_absent(monkeypatch):
    """Render/Smithery inject $PORT; the flag defaults to None so it is honored."""
    monkeypatch.setenv("PORT", "9999")
    assert _http.resolve_port(None) == 9999


def test_port_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    assert _http.resolve_port(None) == 8080


def test_blank_port_env_falls_back(monkeypatch):
    monkeypatch.setenv("PORT", "  ")
    assert _http.resolve_port(None) == 8080


def test_invalid_port_env_raises(monkeypatch):
    monkeypatch.setenv("PORT", "http")
    with pytest.raises(RuntimeError, match="invalid PORT"):
        _http.resolve_port(None)


# === CLI wiring ===


def test_cli_defaults_to_stdio():
    args = mcp_server.build_arg_parser().parse_args([])
    assert args.transport == "stdio"
    assert args.host == "127.0.0.1"
    assert args.port is None


def test_cli_http_args():
    args = mcp_server.build_arg_parser().parse_args(
        ["--transport", "http", "--host", "0.0.0.0", "--port", "9000"]  # noqa: S104
    )
    assert (args.transport, args.host, args.port) == ("http", "0.0.0.0", 9000)  # noqa: S104


def test_main_http_dispatches_to_run_http(monkeypatch):
    seen: dict[str, object] = {}

    def fake_run_http(server, host, port, tool_provider):
        seen.update(
            server=server,
            host=host,
            port=port,
            tool_provider=tool_provider,
        )

    monkeypatch.setattr(_http, "run_http", fake_run_http)
    mcp_server.main(["--transport", "http", "--port", "8123"])

    assert seen["server"] is mcp_server.server
    assert seen["host"] == "127.0.0.1"
    assert seen["port"] == 8123
    assert seen["tool_provider"] is mcp_server.list_tools


def test_main_stdio_does_not_touch_http(monkeypatch):
    """The stdio arm is unchanged: it runs the asyncio stdio server, nothing else."""
    called = {}

    def fake_asyncio_run(coro):
        called["ran"] = True
        coro.close()

    monkeypatch.setattr(mcp_server.asyncio, "run", fake_asyncio_run)
    monkeypatch.setattr(
        _http,
        "run_http",
        lambda *a, **k: pytest.fail("stdio arm must not enter the http path"),
    )
    mcp_server.main([])

    assert called["ran"] is True
