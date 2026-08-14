# Database migration policy

METTLE database migrations are numbered and forward only. The application owns
the expected schema head in `database.LATEST_SCHEMA_VERSION`. Startup applies
missing versions transactionally; readiness fails when the database is
unavailable or the recorded head differs from the candidate.

Schema version 3 is a one-time security migration. It hashes every API-key row
created under the plaintext schema, including values whose plaintext happens to
look like a 64-character digest. PostgreSQL startup serializes migrations with a
transaction-scoped advisory lock, so concurrent workers cannot hash a row twice.

Some schema-version-2 deployments already stored newly created API keys as
SHA-256 digests before version 3 ran. Version 3 necessarily hashes those rows a
second time because the old column has no storage-format marker. Runtime lookup,
deletion, and quota reservation therefore resolve both the current digest and
that migration alias. If both aliases exist as separate rows, authentication
fails closed as ambiguous. New writes store only the current single digest and
refuse a duplicate through either alias.

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
