# METTLE release notes

## [0.3.2]

### Credential schema

Credential schema `1.0` and suite policy `2026-08-12` remain unchanged.
This release repairs package ownership metadata for registry publication.

### Suite policy

Challenge generation, tier ranges, scoring, and credential semantics remain
unchanged from `0.3.1`.

### Public key changes

No signing key or discovery format change is included. Deployed key identity is
verified separately from package publication.

### Compatibility

The package remains compatible with Python 3.10 through 3.14 and exposes the
same seven reviewed MCP tools. The package README now carries the exact
ownership marker required by the Official MCP Registry. Package, API, and
registry versions advance together to `0.3.2`.

### Known limitations

PyPI `0.3.1` remains a functional package release, but its immutable description
lacks the ownership marker now required for Official MCP Registry publication.
Registry consumers should use `0.3.2`. The broader probabilistic, operational,
human review, rights, and independent review limitations documented for
`0.3.1` remain in force.

## [0.3.1]

### Credential schema

Credential schema `1.0` and suite policy `2026-08-12` are unchanged. This is a
distribution and operations hardening release.

### Suite policy

Challenge generation, tier ranges, scoring, and credential semantics are
unchanged from `0.3.0`.

### Public key changes

No signing-key or discovery-format change is included. Deployed key identity is
verified separately from package publication.

### Compatibility

The package remains compatible with Python 3.10 through 3.14. The public wheel
contains the seven reviewed MCP tools and omits the removed automatic solver.
The Official MCP Registry manifest points to this successor package version.

### Runtime and operations

Release distributions now derive timestamps from the source commit and
normalize source-archive ordering, ownership, modes, and timestamps. Two clean
Linux builders and one clean macOS builder must produce byte-identical wheel and
source-distribution files before publication. A read-only Render gate compares
the live API and MCP services with the reviewed repository configuration while
reducing secret values to presence checks.

### Known limitations

METTLE is a probabilistic behavioral gate. A passing credential does not prove
identity, consciousness, safety, autonomy, personhood, or operator
trustworthiness. Destructive production failover and restore drills, human
accessibility review, rights-cleared held-out fairness evaluation, and
independent protocol, cryptographic, privacy, adversarial ML, and bilateral
reviews require evidence outside repository automation.

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
