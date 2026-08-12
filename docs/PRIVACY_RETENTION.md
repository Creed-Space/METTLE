# METTLE privacy and retention contract

## Data minimization

METTLE needs transient challenge state to score a session. It does not need raw
answers for product analytics, fairness reporting, or deprecation telemetry.
Logs and metrics must never contain challenge answers, bearer tokens, API keys,
private keys, signed credentials, webhook secrets, or full database URLs.
The authored public pages do not load third-party analytics. Operational metrics
remain first-party, process-local, aggregate counters.

The held-out evaluation contract accepts only version, aggregate subject class,
suite, expected decision, observed decision, and an optional non-identifying
cohort. Raw text and stable subject identifiers are rejected.

## Enforced application retention

| Data | Authority | Enforced maximum | Deletion behavior |
|---|---|---|---|
| Public quick session and active challenge state | Redis | 30 minutes from session start | Absolute TTL is not extended by updates. Process cache cleanup uses the same window. |
| Authenticated active session and server-held answers | Redis | 5 minutes | Session and answers receive the active TTL. |
| Completed or cancelled authenticated session | Redis | 1 hour | Result state expires automatically. |
| Cached signed authenticated credential | Redis | 1 hour | First signed envelope expires with the completed-session window. The bearer artifact held by a client remains valid until its signed expiry. |
| Presentation challenge | Redis | 60 seconds or first successful use | Successful verification atomically deletes it. |
| Session lock | Redis | 30 seconds | Token-owned release, with TTL as fallback. |
| PostgreSQL session rows | PostgreSQL | Default 24 hours, configurable from 30 minutes to 30 days | Five-minute cleanup loop deletes rows older than `METTLE_PRIVATE_DATA_RETENTION_SECONDS`. |
| PostgreSQL verification and collusion records | PostgreSQL | Default 24 hours, configurable from 1 hour to 30 days | Cleanup deletes rows older than `METTLE_VERIFICATION_RECORD_RETENTION_SECONDS`. |
| Revocation records | PostgreSQL | Until explicit authority lifecycle removes them | Timed private-data purge deliberately preserves them so an old credential is not resurrected. |
| API key and webhook metadata | PostgreSQL | Until explicit revoke or unregister action | Timed private-data purge deliberately preserves authority state. |
| Privacy-minimal evaluation aggregates | Published evidence store | Per release evidence policy | No raw response or identity is permitted in the first place. |

## Data that deletion cannot recall

A client may retain a signed credential, quick badge, public fixture, or release
artifact. Deleting issuer-side session rows cannot erase a bearer copy. A
credential can be rejected online through revocation, but an offline Ed25519
signature remains cryptographically valid as a historical envelope until expiry.

## External systems

Render logs, GitHub Actions artifacts, provider backups, CDN logs, webhook
recipients, error monitoring, and independent review stores have separate
retention controls. Repository tests do not prove those controls. Production
acceptance requires a receipt naming:

1. log retention and access policy;
2. database and Redis backup retention;
3. GitHub artifact retention for evidence;
4. webhook recipient data agreement;
5. a deletion test against the exact deployed candidate.

## Deletion verification

For each release candidate, create synthetic records older and newer than both
configured cutoffs in an isolated database. Run the purge once, confirm that only
old private records disappear, and prove revocation, API key, and webhook records
remain. In production, verify counts and synthetic canaries without inspecting
real participant content.

Working if: application tests demonstrate TTL and purge boundaries, metrics stay
aggregate, logs contain no prohibited payload, and external provider receipts are
listed as open rather than inferred from local code.
