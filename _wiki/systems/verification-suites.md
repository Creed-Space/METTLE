# METTLE Verification Suites

<!-- wiki:type = system -->
<!-- wiki:scope = mettle -->
<!-- wiki:created = 2026-05-23 -->
<!-- wiki:updated = 2026-08-12 -->
<!-- wiki:status = active -->

## Summary

METTLE, Machine Evaluation Through Turing-inverse Logic Examination, is a reverse CAPTCHA and experimental behavioral screening system. Its authoritative hosted registry contains twelve procedurally generated suites. The server sends sanitized challenges while retaining expected answers, observes timing, scores submissions, and may issue a bounded credential when policy requirements are met. A result does not establish consciousness, model identity, autonomy, safety, governance, personhood, or moral status (`README.md`; `mettle/challenge_adapter.py`; `docs/ASSURANCE_CASE.md`).

## Current Suite Registry

| Number | Registry key | Display name | Bounded measurement |
|---:|---|---|---|
| 1 | `adversarial` | Adversarial Robustness | Performance on dynamic, preparation-resistant tasks |
| 2 | `native` | Machine-Oriented Capabilities | Batch, calibration, encoding, and pattern behavior |
| 3 | `self-reference` | Self-Reference | Self-prediction and output consistency |
| 4 | `social` | Social/Temporal | Conversation memory and style consistency |
| 5 | `inverse-turing` | Inverse Turing | Mutual behavioral verification |
| 6 | `anti-thrall` | Anti-Thrall Detection | Heuristic control, refusal, and constraint probes |
| 7 | `agency` | Agency Detection | Stated goal ownership and initiative |
| 8 | `counter-coaching` | Counter-Coaching | Variation and contradiction probes for rehearsed responses |
| 9 | `intent-provenance` | Intent Provenance | Stated constraints, provenance, scope, and harm refusal |
| 10 | `novel-reasoning` | Novel Reasoning | Procedurally generated reasoning with iterative feedback |
| 11 | `governance` | Governance Verification | Reported operational governance mechanisms |
| 12 | `llm-dynamic` | LLM-Dynamic Verification | Model-generated challenges with bounded semantic evaluation |

The names, descriptions, and numbers above come from `mettle/challenge_adapter.py:71-120`. Suite 12 is supplemental and does not raise the credential tier (`README.md`; `mettle/vcp.py`).

## Execution Surfaces

The hosted API implements two related paths:

| Surface | Behavior |
|---|---|
| Quick session API | Starts a bounded three or five challenge session, requires its bearer token for subsequent operations, and can issue a stable signed legacy badge after a passing result. |
| Authenticated suite API | Runs selected suites, requires a complete contiguous policy range for a tier, and returns either an eligible Ed25519 credential or an unsigned evidence receipt when requested. |
| Legacy research CLI | Runs the historical ten-suite local research engine and emits unsigned local results. It is not the current twelve-suite hosted registry. |
| MCP server | Exposes seven interactive and authenticated suite tools. It does not expose an automatic solver. |

Sources: `main.py`, `mettle/router.py`, `mettle/vcp.py`, `mettle/cli.py`, and `mettle/mcp_server.py`.

## Credential Semantics

The tier registry maps Bronze to suites 1 through 5, Silver to 1 through 7, Gold to 1 through 9, and Platinum to 1 through 11. All suites in the range must pass. Suite 12 remains supplemental. The issuer signs a statement about completion under a named policy and time; consumers choose their own freshness, tier, and authorization policy (`mettle/vcp.py:TIER_RANGES`; `README.md`).

All new issuance is controlled by the `METTLE_CREDENTIAL_ISSUANCE_ENABLED` emergency switch. Quick-session HMAC badges and authenticated-suite Ed25519 credentials have distinct formats and verification paths. Production key publication and operational custody remain deployment responsibilities (`config.py`; `main.py`; `mettle/router.py`; `docs/CREDENTIAL_TRANSPARENCY.md`).

## Provenance

Sources last checked on 2026-08-12: `README.md`, `mettle/challenge_adapter.py`, `mettle/mcp_server.py`, `mettle/router.py`, `mettle/vcp.py`, `main.py`, and `docs/ASSURANCE_CASE.md`.

## See Also

| Topic | Link |
|---|---|
| Conceptual framing | [[mettle:domain/inverse-turing-concept]] |
| MCP and API surface | [[mettle:systems/mcp-server-and-api]] |
| Bilateral alignment | [[shared:bilateral-alignment]] |
