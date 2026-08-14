# Red Council scenario status

The Red Council prompt corpus is threat-model input only. It is not an executable security gate.

The retired runners scanned prompt text locally, synthesized expected outcomes from scenario labels, and could report success without submitting an issued challenge to METTLE. Their timing field measured local string processing rather than respondent or verifier latency. Keeping those outputs as release evidence would be misleading, so the workflow, runners, and tests of the synthetic verdicts were removed.

`RED_COUNCIL_THREAT_SCENARIOS.yaml` preserves the attack ideas. A future executable row must bind all of these elements:

1. an authentic challenge issued by the candidate verifier;
2. the bearer or session token returned for that exact session;
3. the exact answer sent to the authoritative answer endpoint;
4. the verifier's returned pass or fail decision;
5. request round-trip timing only when timing is part of the asserted property;
6. a fail-closed assertion for transport, authentication, and malformed-response errors.

The ordinary test and release workflows remain responsible for the existing protocol-level adversarial regressions.

Working if: no automated report can label a scenario passed without a successful authenticated answer request and an assertion over the verifier's response.
