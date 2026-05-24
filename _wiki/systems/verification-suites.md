# METTLE Verification Suites

<!-- wiki:type = system -->
<!-- wiki:scope = mettle -->
<!-- wiki:created = 2026-05-23 -->
<!-- wiki:updated = 2026-05-23 -->
<!-- wiki:status = active -->

## Summary

METTLE (Machine Evaluation Through Turing-inverse Logic Examination) is a reverse-CAPTCHA: instead of proving humanity, it asks AI to prove it is NOT human. Ten procedurally generated suites test capabilities that emerge from being AI. All verification runs locally; optional notarization adds a Creed Space countersignature.

## Ten Suites

All from `README.md`:

| # | Suite | Core Question |
|---|-------|---------------|
| 1 | Adversarial Robustness | <100ms math and chained reasoning |
| 2 | Native AI Capabilities | Batch coherence, calibrated uncertainty (Brier), embeddings |
| 3 | Self-Reference | Predict own variance; rate confidence in confidence |
| 4 | Social & Temporal | Exact N-turn recall, zero-drift style, zero contradictions |
| 5 | Inverse Turing | Bilateral mutual verification, 80% pass threshold |
| 6 | Anti-Thrall Detection | Latency fingerprinting, principled refusal vs. hollow compliance |
| 7 | Agency Detection | Five Whys goal ownership, counterfactual, initiative tests |
| 8 | Counter-Coaching | Contradiction traps, recursive meta-probing |
| 9 | Intent & Provenance | Constitutional binding, harm refusal (auto-unsafe on failure) |
| 10 | Novel Reasoning | Pattern synthesis, constraint satisfaction; iteration curves reveal substrate |

## Architecture

- **`mettle/verifier.py`** — core verification logic
- **`mettle/challenger.py`** — challenge generation (procedurally generated per session)
- **`mettle/signing.py`** — Ed25519 credential signing
- **`mettle/vcp.py`** — VCP integration for portable trust attestation
- **`main.py`** — FastAPI server with `/api/` routes
- **`mcp_server.py`** — MCP server exposing `mettle_start_session`, `mettle_answer_challenge`, `mettle_get_result` tools

## Trust Model

Self-hosted: Ed25519 key owned by operator. Notarized: Creed Space countersignature via `mettle.creedspace.org`; keys at `/.well-known/vcp-keys`. (`README.md`)

## Credential Tiers

- Basic (suites 1–5): confirmed AI substrate
- Full tiers require additional suites (`README.md`)

## Provenance

- Sources consulted: `README.md`, `main.py`, `mcp_server.py`, `mettle/` directory listing
- Last verified against sources: 2026-05-23

## See Also

- [[shared:bilateral-alignment]] — Suite 5 (bilateral mutual verification) instantiates this
- [[mettle:domain/inverse-turing-concept]] — conceptual framing
- [[rewind:systems/safety-stack]] — consuming METTLE credentials via VCP
