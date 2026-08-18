# MCP Server and API

The MCP server exposes three interactive screening tools: start a session, answer a challenge, and get experimental results (`mcp_server.py:65-143`). The first-party auto-solver and its configuration switch were removed (`mcp_server.py`; `tests/test_mcp_server.py`).

Result language reports whether the reverse CAPTCHA passed, its tier, and the signed badge when issued (`mcp_server.py`). Session bearer tokens are required for answer and result operations (`mcp_server.py`; `main.py`).

Passing responses set `verified=true`; the server attaches one stable signed badge with expiry and a revocable identifier (`mettle/models.py`; `mettle/verifier.py`; `main.py`).
