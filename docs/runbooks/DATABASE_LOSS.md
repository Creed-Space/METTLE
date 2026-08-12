# PostgreSQL loss or corruption

**Owner:** data operator. **Secondary:** security incident commander when
revocation or API-key integrity may be affected.

## Trigger

Readiness reports database or schema unavailable, revocation checks fail closed,
migration state differs from the application head, provider integrity alarms
fire, or restored data does not match the expected pre-incident receipt.

## Immediate safety

1. Freeze deployments and schema migrations.
2. Disable new credential issuance if revocation authority cannot be trusted.
3. Do not switch production to SQLite or an in-memory cache.
4. Preserve database provider events, migration version, deployed SHA, and the
   most recent known-good backup receipt.

## Diagnose without mutation

```bash
curl -sS https://mettle.sh/api/health/ready
git show --no-patch --format='%H' <deployed-source-ref>
python3 scripts/check_migrations.py --check
```

Run the migration check only in a checkout of the deployed source. In the
provider console, inspect primary status, storage, connection saturation,
replication, backup completion, and point-in-time recovery range. Do not run a
startup migration against an uncertain database.

The migration and rollback contract is in `../MIGRATIONS.md`. METTLE uses
forward-only migrations and does not destructively downgrade a live schema.

## Recover

1. Restore the selected backup into a new isolated database, never over the only
   production copy.
2. Bind the deployed candidate to that isolated restore and run health, schema,
   row-count, revocation, API-key, webhook, and synthetic session checks.
3. Prove that records created after the recovery point are either replayed from
   an authorized ledger or explicitly declared lost. Never recreate a revocation
   from guesswork.
4. Promote or reconnect through the provider's controlled procedure.
5. Keep issuance disabled until durable revocation checks and schema readiness
   pass from every worker.

## Close with evidence

Retain backup ID and timestamp, restore target ID, migration before and after,
synthetic canary results, authority-record counts, recovery point and data-loss
window, deployed SHA, and the decision that re-enabled issuance.

Working if: production uses PostgreSQL at the expected migration head, authority
records match the accepted recovery point, no revoked credential is resurrected,
and the restore was proven in isolation before promotion.
