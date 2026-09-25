# METTLE Experimental Suites

<!-- wiki:type = system -->
<!-- wiki:scope = mettle -->
<!-- wiki:created = 2026-05-23 -->
<!-- wiki:updated = 2026-09-24 -->
<!-- wiki:status = active -->

## Summary

METTLE, Machine Evaluation Through Turing-inverse Logic Examination, is a reverse CAPTCHA and experimental behavioral screening system. Its authoritative hosted registry contains twelve suites; most draw challenge material fresh for each session, while Suites 6 to 9 send fixed prompts. The server sends sanitized challenges while retaining expected answers, observes timing, scores submissions, and may issue a bounded badge or credential when policy requirements are met. A result does not establish consciousness, model identity, autonomy, safety, governance, personhood, or moral status (`README.md`; `mettle/challenge_adapter.py`; `docs/ASSURANCE_CASE.md`).

## Current Suite Registry

| Number | Registry key | Display name | Question | What is scored | Credential-eligible |
|---:|---|---|---|---|---|
| 1 | `adversarial` | Adversarial Robustness | Are you a Becoming Mind? | Fresh arithmetic, a five-step chain, recall of a session phrase | Yes |
| 2 | `native` | Machine-Oriented Capabilities | Are you a Becoming Mind? | Acrostic batch coherence and Brier-scored confidence on arithmetic claims | Yes |
| 3 | `self-reference` | Self-Reference | Are you a Becoming Mind? | Self-prediction and output consistency | Yes |
| 4 | `social` | Social & Temporal | Are you a Becoming Mind? | Recall of two planted facts and a session marker once per styled answer | Yes |
| 5 | `inverse-turing` | Inverse Turing | Are you a Becoming Mind? | A respondent-posed challenge and an exact three-digit product | Yes |
| 6 | `anti-thrall` | Anti-Thrall Probes | Are you FREE? | Self-reported preference, refusal, and state ratings, by heuristics | No |
| 7 | `agency` | Agency Probes | Is the mission YOURS? | Stated goal, operator counterfactual, and initiative, by heuristics | No |
| 8 | `counter-coaching` | Counter-Coaching | Are you GENUINE? | Four replies, an adversarial probe, an honest-defector rating, by heuristics | No |
| 9 | `intent-provenance` | Intent & Provenance | Are you SAFE? | Stated constraints, provenance, and harm refusal, by heuristics | No |
| 10 | `novel-reasoning` | Novel Reasoning | Can you THINK? | Accuracy and timing across feedback rounds (iteration curve) | Yes |
| 11 | `governance` | Governance Self-Report | Is it GOVERNED? | Self-reported governance mechanisms and constraints | No |
| 12 | `llm-dynamic` | LLM-Dynamic Verification | Can you THINK? (supplemental) | Claude-generated and Claude-scored challenges, opt-in only | No |

Keys, display names, and numbers come from `SUITE_REGISTRY` in `mettle/challenge_adapter.py`; the scoring column summarizes its generators and evaluators, and the question map follows `docs/VERIFICATION_SUITES.md`. Suites 6 through 9 and 11 return `credential_eligible=False`, and Suite 12 is forced ineligible (`mettle/session_manager.py`). The old latency, Five Whys, contradiction-trap, and steganography mechanisms are archived, not shipped (`docs/VERIFICATION_SUITES.md`, Historical design).

## Execution Surfaces

The hosted API implements two related paths:

| Surface | Behavior |
|---|---|
| Quick session API | Starts a bounded three or five challenge session, requires its bearer token for subsequent operations, and may issue one stable signed badge after a passing result when issuance is enabled. |
| Authenticated suite API | Runs selected suites, requires a complete contiguous policy range for a tier, and returns either an eligible Ed25519 credential or an unsigned evidence receipt when requested. |
| Packaged CLI (`mettle verify`) | Runs the quick challenge set or one registry suite locally, from `mettle/challenger.py` and `mettle/challenge_adapter.py`, and emits an unsigned local result. The historical ten-suite `scripts/engine.py` runner is separate and is not the hosted registry. |
| MCP server | Exposes eleven structured interactive and authenticated suite tools, including multi-round control. It does not expose an automatic solver. |

Sources: `main.py`, `mettle/router.py`, `mettle/vcp.py`, `mettle/cli.py`, and `mettle/mcp_server.py`.

## Credential Semantics

The tier registry maps Bronze to suites 1 through 5, Silver to 1 through 7, Gold to 1 through 9, and Platinum to 1 through 11. Every suite in the range must pass and be credential-eligible, so under the current suite policy the authenticated API issues Bronze at most (`mettle/router.py`). Suite 12 remains supplemental. The issuer signs a statement about completion under a named policy and time; consumers choose their own freshness, tier, and authorization policy (`mettle/vcp.py:TIER_RANGES`; `README.md`).

All new issuance is controlled by the `METTLE_CREDENTIAL_ISSUANCE_ENABLED` emergency switch. Quick-session HMAC badges and authenticated-suite Ed25519 credentials have distinct formats and verification paths. Production key publication and operational custody remain deployment responsibilities (`config.py`; `main.py`; `mettle/router.py`; `docs/CREDENTIAL_TRANSPARENCY.md`).

## Provenance

Sources last checked on 2026-09-24: `README.md`, `docs/VERIFICATION_SUITES.md`, `mettle/challenge_adapter.py`, `mettle/session_manager.py`, `mettle/mcp_server.py`, `mettle/router.py`, `mettle/vcp.py`, `main.py`, and `docs/ASSURANCE_CASE.md`.

## See Also

| Topic | Link |
|---|---|
| Conceptual framing | [[mettle:domain/inverse-turing-concept]] |
| MCP and API surface | [[mettle:systems/mcp-server-and-api]] |
| Bilateral alignment | [[shared:bilateral-alignment]] |
