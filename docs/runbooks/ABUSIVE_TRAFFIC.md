# Abusive traffic or credential probing

**Owner:** security operations. **Secondary:** runtime operator and protocol
maintainer when harvesting is involved.

## Trigger

Sustained rate-limit rejection, session or presentation floods, repeated invalid
signatures, challenge-corpus probing, unusual cross-entity synchronization,
resource saturation, or provider abuse alerts.

## Immediate safety

1. Preserve aggregate request, status, latency, dependency, and rate-limit
   evidence. Do not capture raw answers or credentials to improve attribution.
2. Keep request-size, ownership, and rate-limit checks enabled.
3. Prefer provider edge limits and narrow route controls over blocking broad
   populations or regions.
4. Disable issuance if integrity is uncertain, while keeping public status and
   security reporting reachable where practical.

## Diagnose

Compare operation-level volume, status class, request IDs, coarse network source
under the security retention policy, challenge collision metrics, Redis
contention, worker saturation, and pass distributions. Distinguish a traffic
flood from legitimate retries, an accessibility tool, a broken integration, or a
systematic false-rejection appeal.

## Contain and recover

1. Apply the smallest reversible edge rule with expiry and an incident ID.
2. Tighten an existing bounded limit only with evidence and a rollback threshold.
3. Rotate a challenge policy only after measuring replay or reconstruction value;
   do not churn policy to conceal an availability defect.
4. Validate representative allowed clients, CORS origins, MCP use, and retry
   behavior after containment.
5. Remove temporary controls when the trigger falls below the recorded threshold.

## Close with evidence

Retain aggregate volumes, affected operations, saturation point, edge-rule ID and
expiry, false-positive assessment, policy version, exact SHA, and post-control
availability and false-decision checks.

Working if: service integrity and availability recover without storing participant
content, legitimate clients retain a documented path, and every temporary block
has an owner, expiry, and rollback condition.
