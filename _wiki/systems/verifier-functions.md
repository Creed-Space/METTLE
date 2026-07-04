# Verifier Functions

<!-- wiki:type = system -->
<!-- wiki:scope = mettle -->
<!-- wiki:created = 2026-05-23 -->
<!-- wiki:updated = 2026-05-23 -->
<!-- wiki:status = active -->

## Summary

`mettle/verifier.py` contains per-challenge-type `verify_*` functions. Each takes a `Challenge`, the agent's `answer`, and `response_time_ms`; returns a `VerificationResult`. All functions implement anti-harvesting: correct answers are only revealed in the result if the challenge is passed.

## Anti-Harvesting Pattern (`verifier.py:22–30`)

All verify functions follow this pattern:

```python
details = {"correct_answer": correct, "time_ok": time_ok, "received": user_answer}
if passed:
    details["expected"] = challenge.data["expected_answer"]
```

The expected answer is withheld on failure. This prevents a probing agent from submitting wrong answers to extract the answer set. Comment in source: `# SECURITY: Only include expected answer if passed (prevents answer harvesting)`.

## Implemented Verify Functions

### `verify_speed_math` (`verifier.py:8–38`)

Parses answer as `int`. Checks numeric correctness AND `response_time_ms <= challenge.time_limit_ms`. Both conditions required; passing both is the only way to get `details["expected"]`.

### `verify_chained_reasoning` (`verifier.py:41–71`)

Same structure as speed_math. On pass, also reveals `details["chain"]` — the full chain of operations, which serves as an audit trail that the agent followed the correct reasoning path (`verifier.py:60–62`).

### `verify_token_prediction` (`verifier.py:74–80`)

Normalizes both `answer` and `expected` to lowercase. Accepts if the expected token is *contained in* the response (`expected in user_answer or user_answer == expected`). More lenient than exact-match: the agent can produce natural prose as long as it includes the predicted token.

## Result Model

`VerificationResult` (from `mettle/models.py`): `challenge_id`, `challenge_type`, `passed`, `details`, `response_time_ms`, `time_limit_ms`.

## Provenance

- Sources: `mettle/verifier.py:1–80`
- Last verified: 2026-05-23

## See Also

- [[mettle:systems/session-manager-redis]] — session state machine that calls verifier
- [[mettle:systems/verification-suites]] — full ten-suite inventory
- [[mettle:systems/challenge-generation]] — challenge generation that populates `challenge.data`
