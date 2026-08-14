# METTLE credential transparency

## Credential families

| Family | Intended verification | Signature | Default lifetime | Revocation behavior |
|---|---|---|---|---|
| Public quick badge | Online through the issuer | JWT with HS256 | 24 hours | Issuer checks the durable JTI revocation set. No offline third party can validate it without the issuer secret. |
| Authenticated suite credential | Offline signature inspection plus online portable acceptance | Ed25519 | 1 hour | Schema 1.1 portable acceptance requires a fresh issuer-signed status receipt. |
| Presence credential | Online holder presentation only | Ed25519 and holder proof of possession | 1 hour | Generic portable verification rejects the envelope. The issuer checks revocation, audience, holder key, and a one-time presentation challenge. |
| Evidence receipt | Interpretation only | Unsigned | Envelope timestamp only | It is deliberately not a credential and grants no tier. |

Quick badges and Ed25519 credentials are different formats. A consumer must not
infer offline portability from the word “signed” alone.

## Versions and algorithms

* Current and accepted credential schema: `1.1`.
* Current and accepted suite policy: `2026-08-14`.
* Portable issuer signature: Ed25519.
* Content digest: SHA-256 over versioned canonical metadata.
* Quick badge signature: JWT HS256, issuer `mettle-api`.
* Allowed verifier clock skew: at most 30 seconds.
* Legacy, omitted, and unknown credential or suite policy versions fail closed.

Schema `1.0`, suite policy `2026-08-12`, and version-omitting Ed25519 envelopes
were retired as an urgent security exception on 2026-08-14. Signature validity
does not establish that an envelope was issued under the repaired policy. A
relying service must therefore reject those envelopes rather than preserve
compatibility with credentials produced under the solver-exposed policy.

Versioned canonical JSON recursively sorts object keys, uses UTF-8 without ASCII
escaping, removes insignificant whitespace, and emits integral numeric values as
JSON integers. Consumers should use `fixtures/credentials/v1.json` rather than
reconstructing the rule from prose.

## Issuance semantics

A schema `1.1` credential records a bounded session result, tier, passed and
failed suites, difficulty, subject, self-asserted entity if supplied, revocable
identifier, status endpoint, issuance time, expiry, schema, and suite policy.
Authenticated tiers require every suite in the corresponding contiguous range.
The result endpoint atomically caches the first signed credential for a session,
so concurrent retries return the same envelope. Unsigned evidence does not
reserve the issuance slot.

The claim means “this subject met this METTLE policy during this session.” It does
not mean identity, consciousness, safety, autonomy, personhood, moral status,
good intent, or trustworthy operation.

## Public key discovery

`GET /api/mettle/.well-known/vcp-keys` returns:

* `key_id`, `algorithm`, `public_key_pem`, availability, and active status;
* supported credential and suite policy versions;
* a `keys` array containing the active key and any `verify-only` overlap keys.

Production must fail startup if the active signing key is absent, malformed, or
conflicts with a verify-only key that has the same ID. Private keys never appear
in this endpoint, logs, fixtures, or release artifacts.

## Key history

| Key ID | Role | Status | Fingerprint receipt |
|---|---|---|---|
| `mettle-fixture-v1` | Deterministic public compatibility fixture only | Test-only, never production | Public material is embedded in `fixtures/credentials/v1.json` |
| `mettle-vcp-v1` | Configured production key ID in `render.yaml` | Publication not proven from this repository | **Pending production publication receipt** |

This table does not assert that a Render service currently holds or serves the
named production key. The public endpoint, deployed SHA, provider configuration,
and independently computed fingerprint must agree before that row can be marked
active.

## Rotation and emergency revocation

1. Generate the new Ed25519 key outside the repository and store the private key
   in the deployment secret manager.
2. Add the old public key to `METTLE_VCP_VERIFYING_KEYS` under its original ID.
3. Deploy the new private key and unique active key ID together.
4. Verify discovery, issue a new credential, and validate old and new credentials
   from an independent consumer.
5. Retain the old verify-only key until every old credential has expired plus the
   30 second allowance and cache safety margin.
6. Publish the before and after fingerprints, exact SHA, timestamps, and overlap
   receipt.

For compromise, stop issuance first, preserve logs and key metadata, remove the
compromised key from active use, revoke affected JTIs where known, publish an
incident statement, and follow the signing-key compromise runbook. Offline
Ed25519 signatures cannot be recalled by deleting issuer-side session data.

Working if: an independent verifier can select the signed key ID, reproduce all
seven compatibility cases, reject legacy, omitted, and unknown versions, reject
a copied Presence envelope without a live holder presentation, require a fresh
good status receipt for schema `1.1`, and trace every production key status to a
SHA-bound publication receipt.
