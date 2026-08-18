# Integration and Deployment

## Current Flow

1. An authenticated caller creates an interactive session.
2. The server stores answers separately and returns sanitized challenges.
3. The caller submits responses under the session bearer token.
4. The result reports scores and the highest complete tier. With `include_vcp=true`, a qualifying range receives a server-signed Ed25519 credential; partial ranges receive an unsigned evidence receipt (`mettle/router.py`; `mettle/vcp.py`).

The MCP surface exposes `mettle_start_session`, `mettle_answer_challenge`, and `mettle_get_result`. There is no auto-solve tool (`mcp_server.py:65-143`).

The CLI emits an unsigned local verification result and has no auto-solve or self-signing flag. Portable credentials come from the server issuer (`mettle/cli.py`; `README.md`, Quick Start).

## Deployment Boundary

METTLE results alone must not authorize access, trades, deployments, or privileged actions. The current challenges do not prove model identity, trusted execution, autonomy, safety, or governance (`README.md`, Assurance Boundary).

Historical badge verification remains available for old tokens, but current session paths clear badge fields and do not issue new tokens (`main.py:748-760`; `main.py`, Historical Badge Verification Endpoints).
