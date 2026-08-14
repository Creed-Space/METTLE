# Verifier Functions

<!-- wiki:type = system -->
<!-- wiki:scope = mettle -->
<!-- wiki:created = 2026-05-23 -->
<!-- wiki:updated = 2026-08-14 -->
<!-- wiki:status = active -->

## Quick API Verifiers

`mettle/verifier.py` verifies the five quick challenge types. Speed math, chained reasoning, and token prediction require correctness within the server-observed limit. Token prediction uses an exact normalized token match. Instruction following and consistency use type-specific rubrics. Public details report pass state and bounded diagnostics; they never include an expected answer or reasoning chain (`mettle/verifier.py`; `tests/test_verifier.py`).

## Authenticated Suite Verifiers

`mettle/challenge_adapter.py` implements the single-shot suite evaluators. Each evaluator consumes caller answers plus server-held evaluation material and returns a bounded score and details. The production route returns those details only after `SessionManager` performs the authoritative transition. Novel reasoning is evaluated per round and requires complete ordered rounds plus the final accuracy threshold (`mettle/challenge_adapter.py`; `mettle/session_manager.py`; `mettle/router.py`).

Self-report-only suites remain behavioral evidence. They are excluded from credential-bearing tier evidence, as is the model-judged `llm-dynamic` suite. Credential issuance recomputes the complete contiguous tier from durable results rather than accepting a caller's tier claim (`mettle/vcp.py`; `tests/test_security_scan_20260814.py`; `tests/test_vcp.py`).

## Confidentiality Boundary

Expected values remain in server-side challenge state. Both successful and failed responses omit them. This prevents one response from turning the verification endpoint into an answer oracle, while leaving source-aware solving, relay, and semantic-transfer risks as explicit limitations (`mettle/verifier.py`; `mettle/challenge_adapter.py`; `docs/SECURITY_WHITEPAPER.md`).

## Provenance

Sources last checked on 2026-08-14: `mettle/verifier.py`, `mettle/challenge_adapter.py`, `mettle/session_manager.py`, `mettle/router.py`, and `mettle/vcp.py`.
