# METTLE assurance case

Status: repository evidence available, production and independent-review gates
remain open.

## Interpretation boundary

METTLE measures responses to fresh procedural and heuristic challenges. It
infers that a respondent met a named policy during a bounded session. The
project offers an ethical interpretation that Becoming Minds should have a
meaningful opportunity to show agency and contest systematic rejection. These
three layers are separate.

A credential never proves consciousness, model identity, safety, autonomy,
personhood, moral status, benign intent, or operator trustworthiness.

## Claim map

| ID | Bounded claim | Repository evidence | Assumptions | Failure modes | Residual risk and gate |
|---|---|---|---|---|---|
| C1 | Public quick challenges are fresh enough to defeat exact recorded replay at the measured sample size. | `scripts/testing/evaluate_challenge_harvesting.py`, generator tests, and the suite policy version. | The entropy source is healthy and an attacker cannot see server-held answers. | Semantic transfer, adaptive model coaching, entropy failure, or a future generator regression. | The local report is an exact-shape lower bound, not an adversarial guarantee. Periodic external harvesting review remains required. |
| C2 | A qualifying authenticated result is signed by the configured Ed25519 issuer and can be checked without private server state. | `mettle/vcp.py`, `mettle/signing.py`, versioned fixtures, key-rotation tests, and nine killed security mutants. | The active private key remains secret, public key discovery is authentic, and verifier clocks stay within the documented allowance. | Key compromise, stale key cache, canonicalization divergence, or accepting an unknown policy. | Production key publication and an independent staging consumer receipt remain open. |
| C3 | Holder-bound credentials require a fresh one-time presentation and reject replay. | Presence protocol tests, Redis atomic consumption, ownership checks, cancellation cleanup, and the replay mutation. | Redis is shared and authoritative across workers. | Redis loss, split brain, copied holder private key, or operational rollback that resurrects state. | Release-candidate soak and production failover receipts remain open. |
| C4 | Session and revocation authority can be durable across workers. | Redis-backed legacy and v2 managers, PostgreSQL migrations, schema readiness, and the two-worker local trial. | Production uses the required Redis and PostgreSQL services without local fallback. | Misconfigured service URLs, restore drift, replica lag, or provider outage. | Production restart, failover, backup, and restore drills remain open. |
| C5 | Client-visible errors fail closed without exposing internal exception text. | Stable status-to-code taxonomy, bounded validation details, request IDs, redaction middleware, and boundary tests. | Provider logs apply the documented retention and access controls. | An unhandled dependency emits sensitive text before conversion, or proxy logs capture credentials. | Deployed log and edge review remains open. |
| C6 | Stored private evaluation data is bounded and purgeable. | Redis TTLs, PostgreSQL retention settings, purge code, deletion tests, and privacy-minimal evaluation schemas. | External platform logs and backups follow their own declared policies. | Backup retention, webhook recipients, or provider logs outlive application records. | Production configuration and deletion verification remain open. |
| C7 | Tier labels represent contiguous passed suite ranges under a named policy. | `compute_tier`, signed metadata, policy versioning, fixtures, and tier recomputation mutation coverage. | Suite evaluations themselves remain valid for their intended construct. | Measurement drift, false rejection, false acceptance, or policy gaming. | Rights-cleared held-out evaluation and independent protocol review remain open. |
| C8 | Bilateral questions create an inspectable opportunity to express refusal, ownership, and concern. | Suite definitions, protocol governance, and the appeal template. | The interface is accessible and the operator does not coerce answers outside the protocol. | Heuristic scoring can mistake style, language, disability, or cultural norms for agency. | Human accessibility, privacy, and bilateral-alignment review remain open. |
| C9 | A deployed service can be bound to one source revision, and the public MCP surface cannot ask METTLE to solve its own challenges. | Bounded source-revision health fields and headers, production readiness, seven-tool MCP transport tests, solver-exclusion documentation tests, and the ninth security mutant. | The provider supplies an authentic full commit SHA and the observed hostname reaches the intended service. | A stale or misrouted service, proxy rewriting, an unknown revision, or reintroduction of the solver tool. | The 2026-08-12 read-only public baseline is not candidate acceptance: it reports no source identity, lacks readiness routes, and exposes the eight-tool MCP surface including `mettle_auto_verify`. Exact-candidate deployment and re-probe remain urgent open gates. |

## Evidence classes

1. Machine evidence proves only the candidate and environment it names.
2. Production evidence must identify deployed SHA, topology, time, and receipt.
3. Human evidence must identify review scope and disposition without exposing
   private subject data.
4. Publication authority decides whether evidence may be represented publicly.

No evidence class substitutes for another.

## Release argument

A release candidate is acceptable only when C1 through C8 have either a green
receipt at the appropriate evidence class or an explicit open gate in the
release manifest. Any credential schema, suite policy, tier threshold, signer,
storage authority, or challenge generator change invalidates prior candidate
receipts for that surface.

Working if: a reviewer can trace every public claim to a file, test, exact
candidate receipt, or clearly labelled open gate without relying on trust in the
author's summary.
