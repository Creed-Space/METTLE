# Signing and Credentials

<!-- wiki:type = system -->
<!-- wiki:scope = mettle -->
<!-- wiki:updated = 2026-08-14 -->
<!-- wiki:status = active -->

## Current Formats

METTLE issues credentials only through server-owned signers. Quick sessions use the server HMAC badge issuer and earn Bronze or Silver when the quick policy passes. Authenticated suite sessions use Ed25519 and earn Bronze through Platinum only when every suite in the relevant contiguous range passes. Partial, failed, cherry-picked, self-report-only, and LLM-only results cannot mint a tier (`main.py`; `mettle/vcp.py`).

Current issuance uses credential schema `1.1` and suite policy `2026-08-14`. New credentials bind the issuer, session, policy, tier, expiry, revocable identifier, and entity-source marker. Public quick-session entity identifiers are marked self-asserted. Schema `1.0` remains interpretable for already-issued, unexpired credentials, but cannot satisfy the current portable online-status acceptance contract (`mettle/protocol.py`; `mettle/vcp.py`; `docs/CREDENTIAL_TRANSPARENCY.md`).

## Verification and Status

Verification checks the configured issuer key or keyring, key identifier, signed envelope, schema and policy support, recomputed tier, bounded clock skew, expiry, and entity marker. Current portable acceptance also requires an issuer-authenticated status receipt for the credential JTI. Presence credentials require subject-key proof semantics in the JavaScript, Rust, and Python reference verifiers (`mettle/vcp.py`; `mettle/signing.py`; `examples/verify_credential_fixture.js`; `examples/verify_credential_fixture.rs`).

The legacy badge verifier accepts tokens only in a POST body. The replayable credential-in-URL GET route has been removed. Badge verification checks signature, issuer, expiry, identifier, and revocation state (`main.py`; `docs/DEPRECATION_POLICY.md`).

## Governance Boundary

Caller-supplied VCP strings remain unverified metadata. All operational governance flags remain false, and no operator contact or runtime attestation is accepted or returned. Suite names and scores do not promote those values into issuer claims (`mettle/router.py`; `mettle/api_models.py`; `tests/test_attestation_security.py`).

## Provenance

Sources last checked on 2026-08-14: `main.py`, `mettle/protocol.py`, `mettle/signing.py`, `mettle/vcp.py`, `mettle/router.py`, `mettle/api_models.py`, and `docs/CREDENTIAL_TRANSPARENCY.md`.
