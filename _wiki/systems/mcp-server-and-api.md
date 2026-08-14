# MCP Server and API

<!-- wiki:type = system -->
<!-- wiki:scope = mettle -->
<!-- wiki:updated = 2026-08-14 -->
<!-- wiki:status = active -->

## MCP Surface

The packaged MCP server at `mettle/mcp_server.py` exposes seven tools:

| Tool | Purpose | Credential boundary |
|---|---|---|
| `mettle_start_session` | Start a quick interactive session | Returns a per-session bearer token |
| `mettle_answer_challenge` | Submit the caller's answer | Requires the session bearer token |
| `mettle_get_result` | Retrieve the quick-session result | Requires the session bearer token |
| `mettle_list_suites` | List authenticated suite capabilities | Requires the configured API bearer key |
| `mettle_start_v2_session` | Start an authenticated suite session | Requires the configured API bearer key |
| `mettle_verify_suite` | Submit one suite's answers | Requires the configured API bearer key |
| `mettle_get_v2_result` | Retrieve tier evidence and requested credential | Requires the configured API bearer key |

The server intentionally has no automatic solver. A client must answer the challenges itself before a result can reach an issuer. The absence is asserted by `tests/test_mcp_server.py`, `tests/test_documentation_consistency.py`, and the security mutation gate.

## Transports

The same tool set is available through stdio and the bounded Streamable HTTP adapter. The HTTP adapter defaults to loopback, requires an explicit opt-in for a non-loopback bind, accepts POST for the MCP endpoint, caps request bodies, returns stable parse errors, and provides a health endpoint (`mettle/_http.py`; `tests/test_mcp_http.py`).

## API and Result Semantics

Quick sessions use a bearer token minted by `/api/session/start`. Passing sessions can receive one stable signed legacy badge when issuance is enabled and signing is configured (`main.py`).

The authenticated suite API keeps expected answers server-side, computes the highest complete tier, and may return a server-owned Ed25519 credential. Partial or nonqualifying results receive an unsigned evidence receipt when VCP output is requested (`mettle/router.py`; `mettle/session_manager.py`; `mettle/vcp.py`).

Result language reports bounded behavioral evidence. It does not certify consciousness, identity, autonomy, safety, or governance (`README.md`; `docs/ASSURANCE_CASE.md`).

## Provenance

Sources last checked on 2026-08-14: `mettle/mcp_server.py`, `mettle/_http.py`, `main.py`, `mettle/router.py`, `mettle/session_manager.py`, and `mettle/vcp.py`.
