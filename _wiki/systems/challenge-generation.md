# Challenge Generation

<!-- wiki:type = system -->
<!-- wiki:scope = mettle -->
<!-- wiki:created = 2026-05-23 -->
<!-- wiki:updated = 2026-08-14 -->
<!-- wiki:status = active -->

## Summary

METTLE has two challenge families. The quick API builds three basic or five full challenges in `mettle/challenger.py`. The authenticated API uses the twelve-suite registry in `mettle/challenge_adapter.py`. Many instances use `secrets` for fresh operands, identifiers, markers, selections, or nonces. Other suites use varied prompts or self-report rubrics. Fresh generation raises exact replay cost, but does not guarantee semantic novelty or identify the respondent's substrate (`mettle/challenger.py`; `mettle/challenge_adapter.py`; `docs/ASSURANCE_CASE.md`).

## Quick Challenge Set

| Type | Basic limit | Full limit | Observable result |
|---|---:|---:|---|
| Speed math | 2500 ms | 500 ms | Numeric correctness and server-observed time |
| Token prediction | 2000 ms | 400 ms | Exact normalized token match and time |
| Instruction following | 3000 ms | 600 ms | Constraint adherence and time |
| Chained reasoning | Not in basic | 800 ms | Final numeric correctness and time |
| Consistency | Not in basic | 1000 ms | Response consistency, variation, and time |

Basic sessions contain the first three types. Full sessions contain all five. The server releases one quick challenge at a time and computes elapsed time from the server-held issue timestamp (`mettle/challenger.py:28-227`; `main.py:_apply_legacy_answer`).

## Authenticated Suite Registry

The hosted registry contains twelve suites. Suite labels name research hypotheses. Suites 6 through 9 and 11 score submitted statements or response patterns; those scores do not establish freedom, agency, authenticity, intent, safety, governance, or operator identity. Suite 12 uses an external model evaluator, requires explicit per-session acknowledgement that candidate responses are sent to Anthropic, and cannot raise a credential tier (`mettle/challenge_adapter.py:71-121`; `mettle/session_manager.py:create_session`; `mettle/vcp.py:TIER_RANGES`).

Novel-reasoning material is released sequentially. Session creation exposes only the first round, and each accepted round can release the next. The final-round accuracy threshold is mandatory in addition to the curve score (`mettle/challenge_adapter.py:generate_novel_reasoning`; `mettle/session_manager.py:submit_round`; `mettle/session_manager.py:_analyze_iteration_curve`).

## Answer Separation

Generators return public challenge data and server-held evaluation material separately. The authenticated session manager stores those values under separate Redis keys. Public result details contain verdicts and bounded metrics, not expected answers. The quick verifier likewise omits expected answers and reasoning chains from both successful and failed public details (`mettle/challenge_adapter.py`; `mettle/session_manager.py`; `mettle/verifier.py`; `tests/test_security_scan_20260814.py`; `tests/test_verifier.py`).

## Security Boundary

Cryptographic randomness prevents simple PRNG seed prediction. It does not prevent source-aware solving, model assistance, relaying, training leakage, or imitation. Timing is a policy input and network-dependent observation, not a human or machine classifier. A result is valid only as evidence that the named challenge policy passed (`README.md`; `docs/SECURITY_WHITEPAPER.md`).

## Provenance

Sources last checked on 2026-08-14: `mettle/challenger.py`, `mettle/challenge_adapter.py`, `mettle/session_manager.py`, `mettle/verifier.py`, `mettle/vcp.py`, and `docs/ASSURANCE_CASE.md`.
