# METTLE release notes

## [0.4.8]

### Credential schema

Credential schema `1.1` and suite policy `2026-08-14` remain unchanged.
Credential verification now fails closed for cyclic, non-finite, mistyped, and
internally inconsistent signed timing data. Valid credentials remain compatible.

### Suite policy

Challenge generation, scoring, tier eligibility, and the seven reviewed MCP
tools remain unchanged from `0.4.7`. Public answer objects now have explicit
byte, depth, node-count, type, finiteness, and acyclic-structure bounds.

### Public key changes

No signing key or discovery format rotation is included. Key loading now rejects
non-Ed25519 private keys before startup succeeds, and verification rejects
malformed encodings and non-Ed25519 public keys without raising.

### Compatibility

The package remains compatible with Python 3.10 through 3.14. API, CLI, MCP,
and valid credential formats are unchanged. Presence receipts now require
strict integer fields, monotonic coherent timing, and supported action shapes.

### Security

* Bound nested answer JSON against cycles, aliasing, excessive depth and node
  count, oversized scalars, non-string keys, unsupported values, and NaN or
  infinity.
* Reject implicit object stringification, boolean-as-integer answers, negative
  response times, extra consistency answers, and oversized verifier responses.
* Make credential canonicalization and signature verification total over hostile
  values, with permanent regressions and mutation-gate coverage.

### Known limitations

METTLE produces probabilistic behavioral evidence. A passing result does not
prove identity, substrate, consciousness, freedom, agency, safety, governance,
operator identity, or authorization suitability. Relays, source-aware solvers,
model-assisted humans, evaluator error, and imitation remain possible. Human
accessibility, rights-cleared fairness evaluation, independent protocol and
cryptographic review, and destructive recovery drills retain their separate
evidence and authority requirements.

## [0.4.7]

### Credential schema

Credential schema `1.1` and suite policy `2026-08-14` remain unchanged. This
patch does not rotate signing keys or alter credential acceptance semantics.

### Suite policy

Challenge generation, scoring, tier eligibility, and the seven reviewed MCP
tools remain unchanged from `0.4.6`.

### Public key changes

No signing key or discovery format rotation is included. Production key
identity remains independently verifiable through the published discovery
surface and deployment receipts.

### Compatibility

The package remains compatible with Python 3.10 through 3.14. No API, CLI, or
MCP surface changes. The server card and package version advance together to
`0.4.7`.

### Dependencies

* Advance the production resolution to anthropic `0.122.0` and
  typing-inspection `0.4.4`, and the development toolchain to ruff `0.16.3`.
  The reproducibility gate rebuilt byte-identical artifacts on both runners.
* Pin the packaging backend at setuptools `84.0.0`. The exact-pin assertion in
  `test_distribution_backend_is_exactly_pinned` moved in the same commit, so
  the backend still cannot advance without review.
* Advance the pinned GitHub Actions across all five workflows, and the design
  linter to impeccable `3.6.0`.

### Known limitations

METTLE produces probabilistic behavioral evidence. A passing result does not
prove identity, substrate, consciousness, freedom, agency, safety, governance,
operator identity, or authorization suitability. Relays, source-aware solvers,
model-assisted humans, evaluator error, and imitation remain possible. Human
accessibility, rights-cleared fairness evaluation, independent protocol and
cryptographic review, and destructive recovery drills retain their separate
evidence and authority requirements.

## [0.4.6]

### Credential schema

Credential schema `1.1` and suite policy `2026-08-14` remain unchanged. This
patch does not rotate signing keys or alter credential acceptance semantics.

### Suite policy

Challenge generation, scoring, tier eligibility, and the seven reviewed MCP
tools remain unchanged from `0.4.5`.

### Public key changes

No signing key or discovery format rotation is included. Production key
identity remains independently verifiable through the published discovery
surface and deployment receipts.

### Compatibility

The package remains compatible with Python 3.10 through 3.14. No API, CLI, or
MCP surface changes. The server card and package version advance together to
`0.4.6`.

### Maintenance

* Restrict Dependabot's pip ecosystem to declared dependencies. The compiled
  locks were being read as manifests, and the transitive `pydantic-core` pin
  was raised to a release belonging to an unpublished pydantic, which left the
  production lock unresolvable and failed four jobs at dependency install.
  pip-compile owns the closure; CI continues to pip-audit every lock file, so
  transitive advisory coverage is unchanged.
* Read either Font Awesome stylesheet dialect when vendoring the subset fonts.
  Version 7 declares a glyph as the `--fa` custom property rather than
  `content`, and ships the outlines as WOFF2 without TrueType. Both readings
  resolve every icon the site uses to identical codepoints, and the 6.5.1
  output is unchanged byte for byte.
* Advance the vendored browser assets to Font Awesome Free 7.3.1. The
  stylesheet and all four subset fonts are smaller than the 6.5.1 assets they
  replace, and no icon changes identity.

### Known limitations

METTLE produces probabilistic behavioral evidence. A passing result does not
prove identity, substrate, consciousness, freedom, agency, safety, governance,
operator identity, or authorization suitability. Relays, source-aware solvers,
model-assisted humans, evaluator error, and imitation remain possible. Human
accessibility, rights-cleared fairness evaluation, independent protocol and
cryptographic review, and destructive recovery drills retain their separate
evidence and authority requirements.

## [0.4.5]

### Credential schema

Credential schema `1.1` and suite policy `2026-08-14` remain unchanged. This
patch does not rotate signing keys or alter credential acceptance semantics.

### Suite policy

Challenge generation, scoring, tier eligibility, and the seven reviewed MCP
tools remain unchanged from `0.4.4`.

### Public key changes

No signing key or discovery format rotation is included. Production key
identity remains independently verifiable through the published discovery
surface and deployment receipts.

### Compatibility

The package remains compatible with Python 3.10 through 3.14. The public MCP
transport now handles both the documented `/mcp` path and `/mcp/` on the
existing TLS connection, without requiring clients to follow a redirect. The
server card and package version advance together to `0.4.5`.

### Security

* Normalize the bare MCP path inside the ASGI application. Starlette previously
  constructed a slash-canonicalization redirect from the proxy-facing HTTP
  scheme, which produced an `http://` downgrade location at the live Render
  endpoint. Clients that correctly refused the downgrade could not initialize.
* Add a real-socket MCP initialize and `tools/list` regression test using the
  bare path, plus an authentication-boundary assertion that the path never
  redirects before bearer validation.
* Retain the successful `v0.4.4` proxy-identity proof: 60 credential-status
  requests returned HTTP 200 and request 61 returned HTTP 429 despite 61
  distinct caller-supplied `X-Forwarded-For` values. This patch does not change
  that API boundary.

### Known limitations

METTLE produces probabilistic behavioral evidence. A passing result does not
prove identity, substrate, consciousness, freedom, agency, safety, governance,
operator identity, or authorization suitability. Relays, source-aware solvers,
model-assisted humans, evaluator error, and imitation remain possible. Human
accessibility, rights-cleared fairness evaluation, independent protocol and
cryptographic review, and destructive recovery drills retain their separate
evidence and authority requirements.

## [0.4.4]

### Credential schema

Credential schema `1.1` and suite policy `2026-08-14` remain unchanged. This
patch does not rotate signing keys or alter credential acceptance semantics.

### Suite policy

Challenge generation, scoring, tier eligibility, and the seven reviewed MCP
tools remain unchanged from `0.4.3`.

### Public key changes

No signing key or discovery format rotation is included. Production key
identity remains independently verifiable through the published discovery
surface and deployment receipts.

### Compatibility

The package remains compatible with Python 3.10 through 3.14. API and MCP
clients require no changes. The server card and package version advance
together to `0.4.4`.

### Security

* Resolve the Cloudflare visitor identity only after Uvicorn has authenticated
  the Render ingress hop and reduced the forwarded chain to a published
  Cloudflare address. This keeps caller-supplied forwarding headers from
  selecting a quota identity while preventing rotating Cloudflare edges from
  bypassing per-caller budgets.
* Record the failed `v0.4.3` live proxy proof explicitly. The release received
  61 successful credential-status requests because access logs identified
  rotating Cloudflare edges as the caller. Closure therefore moves to the exact
  `v0.4.4` deployment and repeated 61-request proof.

### Known limitations

METTLE produces probabilistic behavioral evidence. A passing result does not
prove identity, substrate, consciousness, freedom, agency, safety, governance,
operator identity, or authorization suitability. Relays, source-aware solvers,
model-assisted humans, evaluator error, and imitation remain possible. Human
accessibility, rights-cleared fairness evaluation, independent protocol and
cryptographic review, and destructive recovery drills retain their separate
evidence and authority requirements.

## [0.4.3]

### Credential schema

Credential schema `1.1` and suite policy `2026-08-14` remain unchanged. This
patch does not rotate signing keys or alter credential acceptance semantics.

### Suite policy

Challenge generation, scoring, tier eligibility, and the seven reviewed MCP
tools remain unchanged from `0.4.2`.

### Public key changes

No signing key or discovery format rotation is included. Production key
identity remains independently verifiable through the published discovery
surface and deployment receipts.

### Compatibility

The package remains compatible with Python 3.10 through 3.14. API and MCP
clients require no changes. The server card and package version advance
together to `0.4.3`.

### Runtime and supply chain

Render proxy trust is restricted to the provider ingress peer network. Uvicorn
now walks forwarded address chains from right to left and rejects
caller-prepended identities instead of trusting every forwarded hop. A live
`0.4.2` probe demonstrated that the wildcard allowed spoofed values to evade
the credential-status limiter, and this release adds a provider-shaped
regression test for both trusted-ingress and untrusted-direct peers.

The exact `0.4.2` API, MCP service, and holder candidate reached Render after
database and Redis URLs were moved to verified TLS with provider-egress-only
datastore ingress. The promotion and post-deploy drift receipts are attached to
the `0.4.2` GitHub Release. This patch preserves those controls and changes only
the proxy trust boundary plus synchronized version metadata.

### Known limitations

METTLE produces probabilistic behavioral evidence. A passing result does not
prove identity, substrate, consciousness, freedom, agency, safety, governance,
operator identity, or authorization suitability. Relays, source-aware solvers,
model-assisted humans, evaluator error, and imitation remain possible. Human
accessibility, rights-cleared fairness evaluation, independent protocol and
cryptographic review, and destructive recovery drills retain their separate
evidence and authority requirements.

## [0.4.2]

### Credential schema

Credential schema `1.1` adds a deterministic revocable identifier, an issuer
status endpoint descriptor, and an explicit self-asserted marker for a supplied
entity identifier to every newly issued Ed25519 credential. Portable acceptance
requires a fresh issuer-signed status receipt. Schema `1.0`, suite policy
`2026-08-12`, and version-omitting envelopes are rejected as an urgent security
exception because they were issued under the solver-exposed policy. Presence
envelopes are rejected by generic portable verifiers and require a fresh live
holder presentation.

### Suite policy

Suite policy `2026-08-14` enforces server-observed time limits in authenticated
sessions, keeps future novel-reasoning rounds and expected answers on the
server, requires the configured final-round accuracy, and excludes supplemental
or self-assertion-only observations from credential tier calculation. Selecting
Suite 12 now requires explicit per-session acknowledgement that candidate
responses are sent to Anthropic. The retired one-step operator commitment is no
longer accepted.

### Public key changes

No signing key or discovery format rotation is included. Discovery lists only
the current schema and suite-policy versions that the verifier accepts.
Production key identity remains a separate deployment receipt.

### Compatibility

This is a deliberate breaking security release. `POST /api/badge/verify` is the
only badge verification operation; the credential-bearing GET path is removed.
Session creation rejects the retired operator commitment field. Public HTTP MCP
requests require a caller-owned key and enforce Host, Origin, content type,
per-caller quota, global authentication budget, bounded principal state, and
concurrency controls. Credential status requests are rate-limited before
database lookup and signing. The seven reviewed MCP tool names and Python 3.10
through 3.14 support remain unchanged. Validation-error details
retain `type`, `loc`, and `msg`, while raw rejected `input` and validator `ctx`
fields are removed to prevent request-content reflection.

### Runtime and supply chain

The `v0.4.1` workflow passed validation, provider drift, reproducibility, and
published source-bound artifacts to PyPI. Public artifact verification and the
installed MCP smoke then passed, but the Official MCP Registry rejected the
124-character discovery description against its 100-character limit. GitHub
Release publication and Render promotion were consequently skipped. This patch
uses a validated 99-character description and runs the pinned Official MCP
publisher's authoritative validation before any future PyPI publication.

The `v0.4.0` tag reached GitHub, but its release workflow was rejected before
any job started because the reusable CI caller did not grant its nested Rust
audit job permission to publish checks. No package, registry entry, GitHub
Release, or Render deployment was created by that attempt. This patch grants
`checks: write` only to the reusable validation caller and preserves the failed
tag rather than moving it.

The API, MCP service, and holder disable mutable-branch auto-deployment. Runtime
state, quotas, revocation, administrator authorization, and retention health use
durable fail-closed authority. Release dependencies are hash locked, release
evidence is rebound to final artifacts, and public PyPI bytes must match the
independent source-built receipt before installation or execution under OIDC
authority. Render drift covers exact secret values and holder managed secret
files, and an in-flight promotion is rollback-bookkept before polling. The
container base is digest pinned, and the public container runs as an
unprivileged fixed user. Production Redis forces certificate and hostname
verification. Database schema migration 3 preserves both current and historical
double-digest lookup aliases, while ambiguity fails closed. Signed legacy badges
become durable in PostgreSQL before Redis publication and cannot be erased or
replaced by compensation.

### Known limitations

METTLE produces probabilistic behavioral evidence. A passing result does not
prove identity, substrate, consciousness, freedom, agency, safety, governance,
operator identity, or authorization suitability. Relays, source-aware solvers,
model-assisted humans, evaluator error, and imitation remain possible. Human
accessibility, rights-cleared fairness evaluation, independent protocol and
cryptographic review, destructive recovery drills, provider TLS configuration,
proxy identity, deployed source identity, and key publication require separate
receipts before their corresponding claims may be closed.

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
