# METTLE deprecation policy

## Notice and compatibility windows

Public API, credential, and suite semantics use these minimum windows:

* an additive deprecation is announced in release notes and OpenAPI metadata;
* a public HTTP operation remains available for at least 180 days and two minor
  releases after notice;
* a credential schema or suite policy remains verifiable through the longest
  signed lifetime plus rotation overlap, even after new issuance stops;
* urgent security removal may use a shorter window only with an incident record,
  migration guidance, and release-authority approval.

`GET /api/badge/verify/{token}` placed credentials in URLs, so it was removed as
an urgent security exception on 2026-08-14. `POST /api/badge/verify` is the only
supported form and carries the credential in a JSON body. The removal is recorded
in the release notes and accepted as an intentional OpenAPI break. No compatibility
shim may reintroduce a credential in a path, query, or redirect URL.

Credential schema `1.0`, suite policy `2026-08-12`, and envelopes omitting those
version fields were also rejected as an urgent security exception on
2026-08-14. Those credentials were issued under the solver-exposed policy and
cannot be safely interpreted as current evidence merely because their issuer
signature remains valid. Schema `1.1` with policy `2026-08-14` is the migration
target. No compatibility shim may bypass its JTI, signed status, identity
binding, or current-policy requirements.

## Telemetry limit

Deprecation evidence may count operation name, status class, coarse client
version when voluntarily supplied, and day-level aggregate volume. It must not
capture badge tokens, challenge answers, response text, raw authorization
headers, IP addresses for product analytics, stable subject IDs, or request
bodies merely to count clients.

Low observed volume is decision input, not proof that no consumer exists.
Maintainers must also inspect public examples, fixtures, MCP tools, documentation,
and known integrations.

## Removal checklist

1. Name the replacement and migration example.
2. Record first notice release and date.
3. Confirm both minimum windows have elapsed.
4. Review aggregate telemetry and known integrations.
5. Mark the OpenAPI break intentional and version the protocol if semantics move.
6. Remove implementation, tests, examples, and documentation together.
7. Publish the exact removal SHA and rollback plan.

Working if: active deprecations are visible in schema and release notes,
security exceptions identify their replacement and removal date, telemetry
contains no participant content, and a removal cannot pass compatibility checks
without an explicit reviewed decision.
