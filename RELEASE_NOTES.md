# METTLE release notes

## [0.3.0]

### Credential schema

Credential schema `1.0` names the schema and suite policy in newly issued
quick badges and Ed25519 credentials. Explicit unknown versions fail closed.
Historical unversioned Ed25519 credentials retain their original verification
behavior until their signed expiry.

### Suite policy

Suite policy `2026-08-12` expands public speed math, token prediction, and
instruction following generators to remove small reusable corpora. Tier ranges
remain contiguous and unchanged.

### Public key changes

The public discovery response now exposes an active Ed25519 key plus optional
verify-only overlap keys. The package does not assert a production fingerprint.
A production key publication receipt remains required before deployment
acceptance.

### Compatibility

The package declares Python 3.10 and later. CI smokes Python 3.10 through 3.14.
Versioned signed fixtures cover Python, JavaScript, and Rust canonicalization,
Unicode, expiry, tampering, and unsupported policy versions. The OpenAPI v1
snapshot has a conservative breaking-change check and a schema-generated client
smoke.

### Runtime and operations

The Vault-backed holder now renews its scoped periodic token from a
service-owned background task, including while signing traffic is idle. Renewal
uses the existing pinned TLS and bounded-response path, retries transient
failures without exposing provider diagnostics, and is cancelled and awaited
during graceful shutdown.

### Known limitations

METTLE is a probabilistic behavioral gate. A passing credential does not prove
identity, consciousness, safety, autonomy, personhood, or operator
trustworthiness. Production Redis and PostgreSQL failover, deployed edge
headers, staging consumer verification, production signing-key publication,
human accessibility review, and independent protocol review require receipts
outside this repository.
