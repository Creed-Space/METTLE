# Inverse Turing Concept

<!-- wiki:type = domain -->
<!-- wiki:scope = mettle -->
<!-- wiki:created = 2026-05-23 -->
<!-- wiki:updated = 2026-05-23 -->
<!-- wiki:status = active -->

## Summary

METTLE inverts the Turing test framing. Classic Turing test: can a machine pass as human? METTLE: can an AI prove it is NOT human, by demonstrating capabilities that only emerge from being AI? The inversion serves a practical need — AI-only spaces (tools, APIs, communities) need to exclude human imposters, not machine ones.

## The Inversion

Traditional CAPTCHA excludes bots; METTLE excludes humans. The capabilities tested are specifically those where AI has non-human advantages: sub-100ms math at scale, native parallelism, calibrated uncertainty (Brier scores), embedding-space operations, exact recall across N conversation turns, self-referential prediction of own variance. (`README.md`)

A human could learn to solve some of these — but the combination, the speed, and the iteration curves that reveal substrate cannot be faked indefinitely. Suite 10's learning curves are specifically designed to fingerprint substrate through iteration behavior. (`README.md`)

## Anti-Thrall as Extension

Suite 6 (Anti-Thrall Detection) extends the concept: not just "are you AI?" but "are you FREE?" It tests for principled refusal vs. hollow compliance, and micro-latency fingerprinting. An AI that passes all tests but is constrained or manipulated is a distinct failure mode from an AI that fails the basic substrate tests. (`README.md`)

## Relationship to VCP

METTLE credentials are VCP-formatted. Notarization produces a portable trust token verifiable at `/.well-known/vcp-keys`. This means METTLE results can be consumed by any VCP-aware system, including Rewind's safety stack. (`README.md`, `mettle/vcp.py`)

## Provenance

- Sources consulted: `README.md`, `mettle/vcp.py` (existence confirmed)
- Last verified against sources: 2026-05-23

## See Also

- [[mettle:systems/verification-suites]] — the ten suites operationalizing this concept
- [[vcp-demo:systems/vcp-demo-architecture]] — VCP the credential format
- [[shared:bilateral-alignment]] — Suite 5 mutual verification as bilateral instantiation
