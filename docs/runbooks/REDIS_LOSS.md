# Redis loss or divergence

**Owner:** runtime operator. **Secondary:** security incident commander when
split brain or unauthorized access is suspected.

## Trigger

`/api/health/ready` reports Redis unavailable, session endpoints return 503,
dependency-error metrics rise, presentation nonces fail, or two workers disagree
about one session.

## Immediate safety

1. Declare the incident and freeze deployments.
2. Do not enable process-local fallback or lower readiness requirements.
3. Preserve provider event IDs, worker logs, request IDs, and the deployed SHA.
4. If workers may have reached different Redis authorities, stop new issuance by
   setting `METTLE_CREDENTIAL_ISSUANCE_ENABLED=false` through the authorized
   provider configuration path, then deploy that configuration once.

## Diagnose without mutation

```bash
curl -fsS https://mettle.sh/api/health/live
curl -sS https://mettle.sh/api/health/ready
curl -sS https://mettle.sh/api/metrics | grep 'dependency="redis"'
```

In the provider console, confirm endpoint, TLS mode, database number, connection
limit, primary identity, replica state, memory pressure, eviction policy, and
recent failover. Compare every worker's redacted configuration fingerprint. Do
not paste the Redis URL into chat, tickets, or shell history.

## Recover

1. Restore the single intended primary or promote through the provider's managed
   failover control. Never create a second writable authority as a workaround.
2. Restart only the METTLE service processes owned by this incident so each opens
   a fresh connection to the same primary.
3. Keep issuance disabled while testing a synthetic session across two workers.
4. Confirm expired sessions, locks, and presentation challenges were not
   resurrected. A lost live session is safer than accepting replayed authority.
5. Re-enable issuance only after readiness, cross-worker state, replay rejection,
   and rate limits pass on the deployed SHA.

## Close with evidence

Retain provider failover ID, before and after primary identity, exact SHA, health
responses, a two-worker synthetic session trace, one consumed presentation
replay rejection, and an explicit statement about live-session loss.

Working if: all workers use one Redis authority, readiness is green, no local
fallback exists, a presentation remains one-time, and issuance resumes through a
single recorded configuration change.
