# Deployment rollback

**Owner:** release operator. **Secondary:** runtime operator and credential
operator when protocol or key material changed.

## Trigger

Post-deploy readiness fails, smoke tests reject the deployed SHA, error or latency
budgets regress, migrations are incompatible, session authority diverges, public
key discovery mismatches, or browser acceptance materially fails.

## Immediate safety

1. Freeze further deploys. `render.yaml` `autoDeploy` is the sole repository
   deployment authority; do not trigger a second hook.
2. Record provider deploy ID, expected and observed SHA, protocol versions, key ID,
   migration state, health, and the last known-good candidate.
3. Disable issuance before rollback when the bad candidate may have issued
   semantically invalid or unverifiable credentials.

## Decide rollback safety

Rollback is unsafe when it would downgrade a database below written data,
resurrect consumed nonces or revoked credentials, remove the only verifier for
unexpired credentials, or reuse a changed key ID. In those cases, prefer a
forward fix or a read-only maintenance deployment.

## Execute

1. Select the exact last known-good provider deploy or a reviewed forward-fix
   commit. Do not rebuild from an unpinned branch.
2. Use the provider's single rollback control. Preserve the failed deploy rather
   than deleting evidence.
3. Confirm `/api/health/live` and `/api/health/ready` return the expected full
   commit in `source_revision`; confirm the `X-METTLE-Source-Revision` response
   header agrees. An `unknown` production revision is a failed deployment. Then
   confirm Redis and database authority, schema head, public keys, and one safe
   synthetic verification flow.
4. List the public MCP tools without invoking one. Require exactly seven tools
   and reject any deployment exposing `mettle_auto_verify`.
5. Run replay rejection and old/new credential verification when the affected
   surface includes sessions or signing.
6. Re-enable issuance through a separate recorded configuration action only after
   acceptance.

## Close with evidence

Retain failed and restored deploy IDs and SHAs, reason, migration and key impact,
health and smoke output, credential impact interval, rollback decision, and
follow-up owner.

Working if: exactly one provider deployment becomes active, the restored SHA is
machine-observed, state authority remains monotonic, and a green local build is
never substituted for deployed acceptance.
