# Backup restore and key-loss recovery

**Owners:** data operator for PostgreSQL, credential operator for signing keys.
These are separate authorities and require separate receipts.

## Scheduled restore drill

1. Select an exact production backup by provider ID and timestamp.
2. Restore into a new isolated database with no production traffic. Never restore
   over the only production instance.
3. Use provider-managed credential injection. Do not paste a DSN into a command,
   transcript, or ticket.
4. Deploy or run the exact candidate that owns the expected migration head
   against the isolated restore.
5. Confirm schema head, table presence, bounded row counts, synthetic canaries,
   revocations, API-key metadata, webhooks, and private-data purge behavior.
6. Verify that old private session rows beyond retention are absent and that
   revocations are not purged.
7. Destroy the isolated restore only after the receipt is complete and explicit
   deletion authority is confirmed.

Local SQLite backup tests characterize application logic only. They are not a
PostgreSQL production restore receipt.

## Point-in-time recovery after incident

Choose a recovery point before corruption and state the data-loss interval.
Reconcile post-point revocations and key actions from an authorized append-only
ledger. Do not infer missing authority records from participant claims. Keep
credential issuance disabled until restored revocation checks fail closed and all
workers observe the promoted database.

## Signing-key loss without compromise

If an Ed25519 private key is irrecoverably lost but not exposed:

1. Keep its public key available as verify-only for unexpired credentials.
2. Generate a new key and unique ID in the secret manager.
3. Publish and validate through `PUBLIC_KEY_PUBLICATION.md`.
4. Do not attempt to reconstruct the private key from a backup outside the
   approved key-management boundary.

If the JWT HS256 secret is lost, existing quick badges cannot be verified and
must expire or be replaced through a new session. If exposure is plausible,
treat loss as compromise.

## Evidence

Retain backup and restore IDs, timestamps, exact SHA, schema head, canary results,
authority-record reconciliation, retention result, destruction receipt, public
key fingerprints, lost-key status, and issuance re-enable decision.

Working if: a clean isolated PostgreSQL restore reaches the expected application
state without resurrecting revoked authority, and key loss produces a new unique
active ID while preserving only the safe public verification history.
