# Database migration policy

METTLE database migrations are numbered and forward only. The application owns
the expected schema head in `database.LATEST_SCHEMA_VERSION`. Startup applies
missing versions transactionally; readiness fails when the database is
unavailable or the recorded head differs from the candidate.

## Safe commands

Inspect without mutation:

```bash
python3 scripts/check_migrations.py --check
```

Apply pending versions deliberately in an isolated restore or controlled
maintenance window:

```bash
python3 scripts/check_migrations.py --apply
```

Both commands read `METTLE_DATABASE_URL`, then emit only the database scheme and
schema status. They never print credentials, host names, or the full DSN.

## Upgrade and rollback contract

Each migration must be idempotent and tested from the oldest supported schema.
Before production application, retain an exact backup identity and candidate
SHA. A source rollback is permitted only when the older candidate understands
the current database head and every field written since deployment. METTLE does
not provide destructive downgrade migrations. When compatibility is uncertain,
restore a proven backup into a new isolated database and reconcile authority
records before promotion.

Working if: a read only check cannot change the migration table, an apply reaches
the exact expected head, repeated apply is idempotent, and unsafe source rollback
is refused rather than paired with a destructive schema downgrade.
