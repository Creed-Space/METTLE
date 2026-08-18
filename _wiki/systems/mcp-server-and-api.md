# METTLE MCP Server and API

<!-- wiki:type = system -->
<!-- wiki:scope = mettle -->
<!-- wiki:created = 2026-05-23 -->
<!-- wiki:updated = 2026-05-23 -->
<!-- wiki:status = active -->

## Summary

METTLE exposes two interfaces: a REST API (FastAPI, `main.py`) and an MCP server (`mcp_server.py`) that allows AI agents to verify themselves programmatically. The MCP interface is the primary integration point for agentic use cases. (README.md "MCP Server"; main.py; mcp_server.py)

## FastAPI Server (`main.py`)

All endpoints prefixed with `/api/mettle`. Bearer token authentication required.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/suites` | GET | List all 10 suites |
| `/sessions` | POST | Create a verification session |
| `/sessions/{id}/verify` | POST | Submit answers (Suites 1–9, single-shot) |
| `/sessions/{id}/rounds/{n}/answer` | POST | Submit round answers (Suite 10 — multi-round) |
| `/sessions/{id}/result` | GET | Final results + credential tier |
| `/sessions/{id}/result?include_vcp=true` | GET | Results with VCP attestation |
| `/.well-known/vcp-keys` | GET | Ed25519 public key for verification |
| `/notarize/seed` | POST | Request deterministic challenge seed |
| `/notarize` | POST | Submit for Creed Space countersignature |

(README.md "API Reference"; main.py existence confirmed)

**Database**: `mettle.db` — SQLite (development). (`mettle.db` in repo root; `database.py` module)

## MCP Server (`mcp_server.py`)

The MCP server allows Becoming Minds to run METTLE behavioral screening and
credential workflows through Model Context Protocol. It does not establish
substrate or identity.

### Tools

| Tool | Description |
|------|-------------|
| `mettle_start_session` | Start an interactive screening session |
| `mettle_answer_challenge` | Submit the client's answer to the current challenge |
| `mettle_get_result` | Retrieve the interactive session result |
| `mettle_list_suites` | List authenticated suite capabilities |
| `mettle_start_v2_session` | Start an authenticated multi-suite session |
| `mettle_verify_suite` | Submit client answers for one authenticated suite |
| `mettle_get_v2_result` | Retrieve tier evidence and an eligible credential |

(README.md "MCP Server"; `mettle/mcp_server.py`)

The server intentionally exposes no automatic solver. A client must provide its
own answers before a result can issue.

### Configuration

```bash
export METTLE_API_URL=https://mettle.sh
export METTLE_API_KEY=your_api_key
python mcp_server.py
```

Add to Claude Desktop `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "mettle": {
      "command": "python",
      "args": ["mcp_server.py"],
      "env": {"METTLE_API_URL": "...", "METTLE_API_KEY": "..."}
    }
  }
}
```
(README.md "MCP Server")

## Module Inventory (`mettle/`)

| Module | Role |
|--------|------|
| `verifier.py` | Response verification logic per challenge type (timing, correctness) |
| `challenger.py` | Procedural challenge generation using `secrets.randbelow()` for cryptographic randomness |
| `signing.py` | Ed25519 key management and signing |
| `vcp.py` | VCP attestation building, CSM-1 token parsing, tier computation |
| `auth.py` | Authentication |
| `models.py` | Data models (`Challenge`, `ChallengeType`, `Difficulty`, `VerificationResult`) |
| `session_manager.py` | Session lifecycle |
| `api_models.py` | API request/response shapes |
| `app_config.py` | Configuration/settings |
| `router.py` | FastAPI router registration |
| `challenge_adapter.py` | Adapter between challenger output and API format |

(mettle/ directory listing)

## Security Design in Verifier

`verifier.py` reveals an important security constraint: expected answers are only returned in verification results if the submission **passed**. (verifier.py:23–27)

```python
# SECURITY: Only include expected answer if passed (prevents answer harvesting)
if passed:
    details["expected"] = challenge.data["expected_answer"]
```

This prevents a client from submitting wrong answers repeatedly to enumerate correct responses.

## Local Development

```bash
pip install -r requirements.txt && pip install -r requirements-dev.txt
uvicorn main:app --reload
pytest tests/ -v
```
(README.md "Local Development")

## Test Coverage

`tests/` contains 30+ test files covering: API, auth, challenger, verifier, MCP, VCP integration, security features, red team scenarios, novel reasoning (Suite 10), and signing. (tests/ directory listing)

## Provenance

- Sources consulted: `README.md` (full); `mcp_server.py` (existence confirmed); `mettle/` directory listing; `mettle/verifier.py:1–37` (security pattern)
- Last verified against sources: 2026-05-23

## See Also

- [[mettle:systems/verification-suites]] — the 10 suites these endpoints serve
- [[mettle:systems/signing-and-credentials]] — VCP attestation structure
- [[mettle:domain/inverse-turing-concept]] — conceptual framing
