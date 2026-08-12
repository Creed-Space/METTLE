# METTLE release checklist

## Candidate identity

* [ ] Tag equals `v` plus the package version.
* [ ] Full source SHA is recorded and matches the checked-out tag.
* [ ] `/api/health`, `/api/health/live`, and `/api/health/ready` report that
  exact SHA, and `X-METTLE-Source-Revision` agrees. Production readiness rejects
  an absent or malformed source identity.
* [ ] Working tree is clean, or every intentional generated artifact is bound by
  a recorded digest and publication remains blocked.
* [ ] No staged or recent-history file exceeds 50 MB.

## Protocol and credentials

* [ ] Credential schema and suite policy match `mettle/protocol.py`.
* [ ] Release notes state schema, policy, public-key, compatibility, and known-limit changes.
* [ ] Python, JavaScript, and compiled Rust fixtures agree.
* [ ] Expiry, unknown-version, tier recomputation, signature, replay, ownership,
  rate-limit, and cancellation mutants are killed.
* [ ] Active and verify-only public keys match the production publication receipt.
* [ ] Independent staging consumer rejects tampering and expiry.

## Build and supply chain

* [ ] Python 3.11 CI and Python 3.10 through 3.14 clean-wheel smokes pass on the SHA.
* [ ] Wheel and source distribution build, install, expose the CLI, and pass `twine check`.
* [ ] Hashed production lock installs cleanly.
* [ ] Runtime and development dependency audits pass.
* [ ] Secret scan, Bandit, Ruff, mypy, Vulture, coverage, npm audit, and frontend checks pass.
* [ ] SBOM, SHA256SUMS, build provenance, curated notes, and
  `RELEASE-MANIFEST.json` are attached.

## Runtime and operations

* [ ] Migration upgrade, schema readiness, isolated backup restore, and restart pass.
* [ ] Two-worker legacy and authenticated session authority passes.
* [ ] Redis loss, recovery, and failover receipts identify the SHA and topology.
* [ ] Holder soak, bounded load, production CORS, trusted host, proxy identity,
  security headers, and post-deploy smoke pass.
* [ ] The public MCP handshake exposes exactly seven tools and does not expose
  `mettle_auto_verify`; no tool invocation is needed for this release check.
* [ ] Rollback target and signing-key incident procedure are rehearsed.

## Product, privacy, and governance

* [ ] Browser routes, responsive sizes, keyboard, focus, zoom, reduced motion,
  captions, transcript, automated accessibility, performance, cache, links, and metadata pass.
* [ ] Human visual, caption, assistive-technology, and device review receipts exist.
* [ ] Retention and deletion receipts cover application and provider systems.
* [ ] Held-out fairness data is rights-cleared, privacy-minimal, sufficient, and reviewed.
* [ ] Independent cryptography, adversarial ML, accessibility, privacy, and bilateral dispositions are complete.
* [ ] Publication authority has approved the exact claims and limitations.

## Release receipt

Record tag, source SHA, deployed SHA, workflow URLs, artifact hashes, protocol
versions, key fingerprints, review receipts, rollback candidate, open gates, and
release-authority identity in one append-only receipt.

Working if: no checkbox is satisfied by a different SHA or evidence class, every
open checkbox is labelled as a gate, and the release manifest makes limitations
visible to consumers.
