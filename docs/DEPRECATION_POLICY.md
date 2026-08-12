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

The current deprecated `GET /api/badge/verify/{token}` form puts a credential in
URLs and is replaced by `POST /api/badge/verify`. Its proposed earliest removal
date is **2027-02-12**. Removal is not authorized until a release note confirms
the date, privacy-preserving usage evidence is reviewed, and the OpenAPI breaking
change is explicitly accepted.

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

Working if: deprecated operations are visible in schema and release notes,
telemetry contains no participant content, and removal cannot pass compatibility
checks without an explicit reviewed decision.
