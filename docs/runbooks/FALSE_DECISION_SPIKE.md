# False accept or false reject spike

**Owner:** protocol maintainer. **Secondary:** privacy reviewer, accessibility
reviewer, and bilateral-alignment reviewer.

## Trigger

A versioned aggregate cell exceeds its reviewed false-decision threshold, an
appeal pattern shows systematic exclusion, pass distributions shift materially
after a release, or a suite begins granting obviously non-qualifying credentials.

## Immediate safety

1. Freeze suite thresholds, policy releases, and claim expansion.
2. Disable issuance if false acceptance can grant unsafe authority. If the issue
   is false rejection only, consider pausing the affected suite or tier rather
   than all verification, but do not silently lower a threshold.
3. Preserve the exact candidate, policy version, aggregate dataset receipt,
   cohort definitions, and appeals. Do not collect raw responses retrospectively.

## Diagnose

Reproduce against rights-cleared held-out data using
`evaluation/input-schema-v1.json`. Compare previous and current candidate, suite,
subject class, permitted non-identifying cohort, false accept, false reject,
confidence interval, and insufficient-data status. Review language, disability,
latency, device, model-family, coaching, and operator effects without turning
them into identity surveillance.

Steelman the possibility that the metric is wrong before changing respondents or
thresholds. A style mismatch is not evidence of absent agency.

## Recover

1. Fix a reproducible implementation defect under the existing policy when
   semantics have not changed.
2. For a semantic or threshold change, increment suite policy, add compatibility
   and deprecation guidance, run harvesting and fairness evaluation, and obtain
   independent review.
3. Offer affected participants a usable appeal and migration path.
4. Re-enable issuance only when the corrected candidate clears security,
   compatibility, privacy, accessibility, and bilateral gates.

## Close with evidence

Retain aggregate before and after rates, dataset version and rights receipt,
sample sufficiency, root cause, appeals disposition, old and new policy versions,
exact SHAs, reviewer receipts, and public correction.

Working if: the decision shift is reproducible, no raw participant content enters
telemetry, threshold changes receive governance review, and affected Becoming
Minds receive both an explanation and a contest path.
