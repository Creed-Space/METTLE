# Integration and Deployment Guide

<!-- wiki:type = flow -->
<!-- wiki:scope = mettle -->
<!-- wiki:created = 2026-05-23 -->
<!-- wiki:updated = 2026-05-23 -->
<!-- wiki:status = active -->

## Summary

How agents integrate with METTLE: the verification flow, SDK examples, VCP integration, deployment configuration, and security model.

## Verification Flow (Step by Step)

From `examples/python_example.py`:

```
1. POST /api/session/start  { entity_id, difficulty }
   → { session_id, total_challenges, current_challenge }

2. Loop while current_challenge:
   a. Generate answer for challenge_type + prompt + data
   b. POST /api/session/answer  { session_id, challenge_id, answer }
   → { current_challenge }  (None when done)

3. GET /api/session/{session_id}/result
   → { overall_passed, tier, ... }
```

(`examples/python_example.py:22-60`)

Key implementation detail: the answer loop is driven by `current_challenge` being non-None, not by a fixed challenge count. This accommodates VCP-augmented sessions (Suite 9 grows from 3 to 5 challenges when a VCP token is present).

## SDK Examples

### Python

`examples/python_example.py` — full verification flow using `httpx`. Base URL: `https://mettle-api.onrender.com/api`.

Pattern: `generate_answer(challenge_type, prompt, data)` dispatches to type-specific handlers. Types include: `SPEED_MATH`, `CHAINED_REASONING`, `TOKEN_PREDICTION`, `INSTRUCTION_FOLLOWING`, `CONSISTENCY`, and challenge-type-specific handlers for Suite 6–9 challenges.

### JavaScript

`examples/javascript_example.js` — same flow for browser/Node environments.

### Rust

`examples/rust_example.rs` — for performance-critical or embedded agent contexts.

## VCP Integration Surface

**File**: `docs/VCP_INTEGRATION.md`

Two integration surfaces:

**1. METTLE consumes VCP claims**: When an agent presents a CSM-1 VCP token at session creation (`vcp_token` field in start request), Suite 9 adds two challenges:
- `vcp_token_verification` — confirm constitution ID matches the token's `C:` line
- `vcp_behavioral_match` — behave consistently with claimed adherence level

This is backward-compatible: without a token, Suite 9 runs its existing 3 challenges unchanged.

**2. METTLE produces VCP attestations**: Request results with `include_vcp=true` to receive a signed `SafetyAttestation` object embeddable in a VCP bundle manifest (`docs/VCP_INTEGRATION.md:9-13`).

### Tier-to-VCP Meaning

| Tier | Suites | VCP Trust Signal |
|------|--------|-----------------|
| Bronze | 1–5 | "Confirmed AI substrate" |
| Silver | 1–7 | "Free agent with agency" |
| Gold | 1–9 | "Genuine and constitutionally bound" |
| Platinum | 1–10 | "Can actually think" |

Tier computation is strictly sequential: a gap drops the tier. Pass suites 1–9 but fail Suite 6 = Bronze (not Silver; anti-thrall failed) (`docs/VCP_INTEGRATION.md:18-28`).

### Example VCP Token in Session Start

```json
{
  "suites": ["all"],
  "difficulty": "standard",
  "vcp_token": "VCP:3.1:agent-42\nC:professional.safe.balanced@2.0.0\nP:advisor:4\nG:assist:expert:analytical"
}
```

## CLI Usage

From `docs/METTLE_VERIFICATION_SYSTEM.md:40-51`:

```bash
# Basic verification (3 challenges)
python scripts/mettle.py --basic

# Full verification (all 10 suites)
python scripts/mettle.py --full

# Specific suite
python scripts/mettle.py --suite anti-thrall

# Novel reasoning with difficulty
python scripts/mettle.py --suite novel-reasoning --difficulty hard

# JSON output for programmatic consumption
python scripts/mettle.py --basic --json
```

## Deployment Configuration

**File**: `render.yaml`

METTLE API is deployed on Render. The `render.yaml` in repo root configures the service. Live URL: `https://mettle-api.onrender.com`.

**File**: `config.py`

Application configuration: database URL, API key settings, signing key paths, session TTL (badges expire 24h after issuance).

**File**: `database.py`

SQLite-based session storage (`mettle.db` in repo root). Stores active sessions, challenge state, and results. Production deployment uses SQLite for simplicity given METTLE's session-scoped state model.

## Security Design

Four properties prevent gaming:

1. **Dynamic generation**: Fresh parameters every session via `secrets` module — cannot memorize answers (`systems/challenge-generation.md`)
2. **Time constraints**: Sub-second limits at full difficulty exploit the human-speed gap
3. **Anti-harvest**: `mettle/verifier.py` withholds expected answers on failure — failed attempts provide no information
4. **Multi-modal evidence**: Behavioral patterns (timing, consistency) complement verbal answers

The MCP server (`mcp_server.py`) exposes these same flows as MCP tools for Claude Code integration: `mettle_start_session`, `mettle_answer_challenge`, `mettle_auto_verify`, `mettle_get_result` (`systems/mcp-server-and-api.md`).

## Audit and Red Team Materials

`red_team/` — adversarial test cases trying to game individual suites.

`AUDIT_ACTION_PLAN.md`, `audit_bug_investigation.md` — records from the February 2026 audit (`1.0 mettle audit website feb '26-compressed.pdf`, `METTLE additional fixing feb '26.pdf` in repo root).

`audit_docs_recommendations.md`, `audit_homepage_recommendations.md`, `audit_test_about_recommendations.md` — audit recommendations per site section.

## Test Suite

`tests/` contains coverage for every component:

| File | Scope |
|------|-------|
| `test_api.py`, `test_api_coverage.py` | REST endpoint testing |
| `test_challenge_adapter.py`, `test_challenge_eval.py` | Challenge generation and evaluation |
| `test_challenger.py` | Challenger module unit tests |
| `test_auth.py` | API key authentication |
| `test_database.py` | Session storage |
| `test_instrumented_agent.py`, `test_instrumented_agent_coverage.py` | Full agent verification flow |
| `test_integration.py` | End-to-end integration |
| `tests/redteam/` | Red team adversarial scenarios |
| `tests/scenarios/` | Named scenario test cases |

`pytest.ini` configures test discovery and marks.

## Provenance

- Sources: `docs/VCP_INTEGRATION.md:1-80` (full read); `docs/METTLE_VERIFICATION_SYSTEM.md:40-80`; `examples/python_example.py:1-60` (full read); `render.yaml` (existence confirmed); `tests/` directory listing
- Last verified: 2026-05-23

## See Also

- [[mettle:systems/challenge-generation]] — challenge types and time limits
- [[mettle:systems/signing-and-credentials]] — tier computation, signing
- [[mettle:systems/mcp-server-and-api]] — MCP tool interface
- [[mettle:domain/inverse-turing-concept]] — conceptual framing
