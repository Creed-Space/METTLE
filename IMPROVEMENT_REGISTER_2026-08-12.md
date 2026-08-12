# METTLE comprehensive improvement register

Date: 2026-08-12

Working branch: `codex/mettle-all-improvements-20260812`

Base commit: `ab805a70b65f8978297db60c394062eebc0055f3`

Candidate state: an isolated working tree derived from the base commit. The source
candidate is locally validated and is bound by
`output/evidence/candidate-working-tree-2026-08-12.json`. It is intentionally not
yet an immutable commit because the authored frontend and media require Nell's
explicit visual approval before staging or committing. No deployment, release, or
publication is claimed.

This register covers product correctness, security, resilience, privacy, API design,
operations, supply chain, testing, packaging, performance, accessibility,
documentation, governance, and maintainability. Evidence classes stay separate:
source checks do not prove runtime behavior; local runtime checks do not prove
production behavior; machine checks do not replace human, rights, publication, or
operational authority.

## Exact-candidate local evidence

| Gate | Result |
|---|---|
| Python test and coverage gate | 1,910 passed on Python 3.12.13 with no reruns; deterministic statement coverage 90.08 percent across 8,263 statements |
| MCP compatibility | MCP 2.0.0 stdio and Streamable HTTP handshakes expose exactly seven tools; the automatic solver surface is absent |
| Security mutation gate | Nine focused security mutants killed; report at `output/evidence/security-mutation-2026-08-12-final.json` |
| Runtime resilience | Redis startup, loss, recovery, failover, two-worker legacy-session behavior, PostgreSQL restore, holder soak, and service-owned idle Vault token renewal exercised locally; reports under `output/evidence/` |
| Browser and accessibility | 17 responsive, axe, keyboard-flow, recovery, reduced-motion, and media tests passed, followed by the performance budget: LCP 140 ms, INP 0, CLS 0.00180, 528,681 critical bytes, and zero initial video bytes. Human review remains separate |
| Packaging and supply chain | Wheel and source distribution passed Twine and clean-install smoke; the Python 3.11 production lock passed `pip check`; the SBOM contains all 44 locked components, 18 direct dependencies, and no dangling references; the MCP container passed seven-tool stdio and HTTP handshakes |
| Static quality | Ruff lint and format, mypy across 124 files, Vulture, broad Bandit, the audited 248-file secret scan, three Python dependency audits, npm audit, JavaScript, static-site, OpenAPI, fixtures, and Impeccable all pass |
| Candidate identity | The working-tree manifest hashes every intended source path. Hosted CI and release evidence still require an immutable committed SHA |

## Candidate-unbound public baseline

Read-only probes on 2026-08-12 are retained at
`output/evidence/public-deployment-baseline-2026-08-12.json` and
`output/evidence/public-mcp-baseline-2026-08-12.json`. The read-only remote
repository receipt is at `output/evidence/public-github-baseline-2026-08-12.json`.
They describe current external state, not this working-tree candidate, and
therefore cannot clear an exact-candidate gate.

* The live API health route returns version 0.3.0 and production environment, but
  reports no source revision. The candidate's liveness, readiness, and metrics
  routes are absent from the live deployment.
* The live MCP handshake exposes eight tools, including `mettle_auto_verify`. The
  candidate exposes seven tools and removes the server-side solver path.
* The live homepage still makes categorical substrate, agency, and safety claims
  that the candidate replaces with bounded measurement language.
* The observed live edge accepted the configured origin, denied an unlisted CORS
  origin, and rejected an untrusted host. These positive baseline observations
  remain candidate-unbound and must be repeated after deployment.
* Remote `main` remains at the candidate's base SHA. Its latest CI run is green,
  but the duplicate deploy-hook workflow is still active there. No hosted workflow
  has run against the uncommitted candidate.
* A separate nonvisual remediation is prepared on
  `codex/mcp-auto-solver-hotfix-20260812`, derived from the same base. Its local
  manifest identity is
  `working-tree-sha256:114604f5bbd4d9005cabb9dc25c79c10a77f6de7b0d7da9f100a5f38bdeb62ec`.
  It removes the solver, repairs the MCP 2 clean-build failure, pins the MCP lock
  in Docker and CI, and removes the duplicate deploy hook. It remains unstaged,
  uncommitted, unpushed, and undeployed pending authority.

## Authoritative execution reconciliation

The status vocabulary is deliberately strict:

* **Local complete** means the repository implementation and appropriate local
  machine proof are complete.
* **Local complete plus external gate** means all locally producible work is
  complete, with the named production, hosted, credential, or independent receipt
  still required.
* **External or human programme** means legitimate completion depends on evidence,
  authority, rights, participants, or environments outside this checkout.
* **No-op justified** means investigation found no evidence-based change that would
  improve the requested outcome.

| # | Status | Implemented result and retained evidence | Remaining nonlocal gate |
|---:|---|---|---|
| 1 | Local complete plus external gate | Unified CI and release workflows cover tests, coverage, lint, typing, dead code, package build, clean install, dependency audits, secrets, frontend, OpenAPI, fixtures, SBOM, and adversarial checks. The local candidate passes the full suite. | Hosted CI must pass on the immutable committed SHA after visual approval. |
| 2 | Local complete plus external gate | Production configuration rejects missing VCP signing material, checks key identity, publishes verifier material, supports a versioned keyring, and has rotation and mismatch tests. | Confirm the live private-key custody, active key ID, and published public-key fingerprint without exposing secret material. |
| 3 | Local complete plus external gate | Redis is authoritative for sessions, ownership, replay, rate limits, revocation coordination, and the legacy quick path; PostgreSQL migrations and restore tooling are tested. Two-worker behavior is locally proven. | Preserve restart, failover, split-brain, and database restore receipts from the production topology. |
| 4 | Local complete plus external gate | The candidate removes the duplicate GitHub deploy-hook workflow. `render.yaml` becomes the sole repository deploy authority, with deployment governance and rollback documentation. | The workflow remains active on remote `main` until the approved candidate lands. Then verify the live Render blueprint has automatic deployment enabled and prove one commit creates exactly one deployment. |
| 5 | Local complete plus external gate | The failure-prone duplicate deploy workflow no longer exists. Provider observation, failure classification, health checks, and rollback expectations are explicit in release and runbook contracts. | Observe a failed or rolled-back staging deploy and retain provider receipts before claiming operational acceptance. |
| 6 | Local complete plus external gate | The exact CI-scope detect-secrets scan completes locally against the reviewed baseline without weakening detection. | Repeat in a clean hosted checkout on the immutable SHA. |
| 7 | Local complete plus external gate | Versioned positive and negative credential fixtures verify independently in Python, JavaScript, and Rust, including tampering, Unicode, expiry, and key selection. | Issue and verify one staging credential through an independently owned consumer, retaining SHA and key fingerprint. |
| 8 | Local complete plus external gate | Local Redis resilience and failover harnesses cover startup loss, mid-session loss, recovery, stable 503 behavior, and shared-state continuity. Reports are retained under `output/evidence/`. | Run the same exact-candidate trial against the production Redis topology. |
| 9 | Local complete plus external gate | Holder-service soak, bounded payload, policy denial, key isolation, timeout, and log-safety paths have local tests and a retained soak report. | Repeat under production Transit or HSM custody, rotation, restart, and representative sustained traffic. |
| 10 | Local complete plus external gate | CORS allowlists, trusted-host enforcement, proxy identity, request IDs, redaction, and security headers are configured and tested at the application layer. | Probe every public origin and header at the deployed edge after Render or CDN transformations. |
| 11 | Local complete | CI builds wheel and source distribution, validates metadata and README rendering, and exercises installed entry points. | None beyond item 1's hosted exact-SHA gate. |
| 12 | Local complete | Development requirements declare pinned build and Twine tooling; packaging metadata uses an SPDX license expression. | None. |
| 13 | Local complete | Production, development, and MCP dependency sets are audited, and npm audit covers the frontend toolchain. | Repeat automatically on dependency-update pull requests. |
| 14 | Local complete | GitHub workflow permissions are job-scoped; only the issue-creation job receives write authority and unused security-event write access is gone. | None. |
| 15 | Local complete | Python 3.11 is release-authoritative, Python 3.12 is a compatibility lane, package metadata declares the supported range, and the Python 3.11 production lock is container-validated. | Hosted matrix execution on the immutable SHA. |
| 16 | Local complete | Clean environments install the built wheel, run CLI help and a deterministic local receipt, verify server-only dependencies are absent, and smoke the MCP extra separately. | Hosted matrix execution on the immutable SHA. |
| 17 | Local complete | Readable direct requirements are paired with hashed production and MCP locks; deployment installs the exact production lock. | Dependency updates still require review before merge. |
| 18 | Local complete | Dependabot schedules grouped Python, npm, and GitHub Actions updates under the normal CI and review gates. | Ongoing review of generated pull requests. |
| 19 | Local complete plus external gate | Release workflows generate CycloneDX SBOMs, finalize a lock-complete dependency graph, build checksums and a release manifest, and preserve source identity. | Signed hosted provenance and published release attachments require an immutable SHA and release authority. |
| 20 | Local complete | Workflow concurrency groups prevent superseded CI and release jobs from racing; deployment itself has one authority. | Confirm provider-side behavior during item 4's live proof. |
| 21 | Local complete plus external gate | Health, safe-flow, deployed-SHA, rollback, and evidence-retention contracts are implemented in code, release checklists, and runbooks. Health fields and `X-METTLE-Source-Revision` expose only a valid full SHA; production readiness fails when identity is unknown. | Deploy one immutable candidate, match every health response and header to its SHA, and retain the provider receipt. The current public baseline fails this gate. |
| 22 | Local complete plus external gate | Versioned schema migrations, advisory locking, schema checks, idempotent startup, upgrade tests, and local PostgreSQL backup/restore proof are present. Downgrade policy is documented as restore-based. | Run upgrade and restore against a production-shaped database backup. |
| 23 | Local complete | The legacy quick-session path uses Redis rather than process-local session authority. A two-worker trial proves cross-worker start, answer, and result continuity. | Production multi-worker rollout remains an operator decision under item 3. |
| 24 | Local complete plus external gate | Structured metrics cover availability, latency, datastore failures, signing failures, rate limiting, pass distributions, and cleanup without recording answers or bearer material. Error-budget targets are documented. | Establish production baselines and alerts from privacy-reviewed aggregate telemetry. |
| 25 | Local complete plus external gate | Bounded request IDs propagate through API and datastore boundaries; injection is rejected and secrets, answers, tokens, credentials, and database credentials are redacted. | Inspect deployed proxy and provider logs to confirm end-to-end preservation and redaction. |
| 26 | Local complete plus external gate | A bounded-load harness reports throughput, p50, p95, p99, saturation, rejection, Redis contention, and signing behavior; a local report is retained. | Repeat at representative production topology and edge latency. |
| 27 | Local complete | Shared protocol helpers and deterministic tests cover issuance, expiry, nonce, session, and bounded clock-skew points immediately before, at, and after each boundary. | None. |
| 28 | Local complete plus external gate | New issuance selects the active key; a versioned keyring verifies unexpired old credentials during overlap and rejects retired keys. Runbooks cover emergency response. | Exercise rotation with production key custody and public-key publication. |
| 29 | Local complete plus external gate | The harvesting evaluator measures collisions, generator diversity, repeated-query value, adaptive behavior, and corpus reconstruction risk; local evidence is retained. | Establish rotation triggers from larger rights-cleared adversarial datasets and ongoing production aggregate evidence. |
| 30 | External or human programme | Dataset schema, evaluation tooling, policy-version capture, aggregate metrics, review rules, and empty evidence boundaries are implemented. No fabricated fairness claim is made. | Obtain rights-cleared held-out Becoming Mind and human-assisted cohorts, run the evaluation, and complete ethics and threshold review. |
| 31 | Local complete plus external gate | Playwright covers the home, test, documentation, and about routes at desktop, tablet, and mobile widths, including navigation, focus, challenge flow, error recovery, and result rendering. | Human device and browser review of the approved visual candidate. |
| 32 | Local complete plus external gate | Axe, keyboard navigation, focus visibility, zoom and reflow, reduced motion, landmarks, names, and live-status behavior are automated in CI. | Human assistive-technology and keyboard-only acceptance. |
| 33 | Local complete plus external gate | Caption and narration consistency is tested; transcript, controls, poster fallback, and keyboard behavior are covered locally. | Human language, caption-timing, audio, and device review. |
| 34 | Local complete plus external gate | A checked performance budget covers LCP, INP, CLS, critical bytes, and deferred video. The explainer does not transfer before user intent. | Confirm production cache and edge performance under representative devices and networks. |
| 35 | Local complete plus external gate | Static assets are content fingerprinted; HTML and manifests revalidate while immutable assets receive long caching. Rollback-compatible references are checked. | Verify actual CDN and Render cache headers after deployment. |
| 36 | Local complete plus external gate | CSP, framing, content-type, referrer, permissions, HSTS policy, and related headers are source-tested. | Validate the final header set at every deployed edge and redirect. |
| 37 | Local complete | One bounded static-site checker validates links, routes, assets, poster, manifest, sitemap, canonical, social metadata, and FastAPI mount semantics. XML input is size-bounded and rejects DTD or entities. | None. |
| 38 | Local complete | Canonical URLs, Open Graph data, sitemap, robots directives, manifest icons, JSON-LD, and production hostname consistency are machine-validated. | Human search-preview inspection is optional publication QA. |
| 39 | Local complete | Footer updates use an idempotent tested Python script with exact HTML scope; the workflow stages only intended paths. | None. |
| 40 | No-op justified | Characterization identified large modules but no extraction seam whose benefit outweighed churn and regression risk. Existing boundaries were extended instead. | Reconsider only when a concrete change is blocked by module coupling. |
| 41 | Local complete | Risk-focused tests cover key loading, process cleanup, Redis faults, parsing, expiry, CLI usage, deployed source identity, MCP transports, SBOM finalization, migration, issuance boundaries, and every procedural math branch. Global coverage is 90.08 percent. | None. |
| 42 | Local complete | A bounded mutation harness mutates temporary copies only. Nine security invariants cover MCP solver exclusion, policy versions, tier recomputation, Ed25519 verification, replay, rate-limit boundaries, ownership, and cancellation cleanup. | Broaden only when new high-value invariants are introduced. |
| 43 | Local complete | Cancellation and timeout tests prove quota rollback, nonce consumption rules, Redis and HTTP cleanup, subprocess bounds, and no reusable half-completed authorization state. | Production soak remains under items 8 and 9. |
| 44 | Local complete | Stable client error codes, redacted messages, request correlation, fail-closed datastore mapping, and structured internal causes are implemented and tested. | Inspect deployed provider error rewriting under item 10. |
| 45 | Local complete | Session start, answer submission, result retrieval, credential issuance, key rotation, and retry behavior have explicit idempotency rules and tests; deploy retry authority is documented. | Provider webhook behavior is covered by the live deployment proof. |
| 46 | Local complete | A reviewed OpenAPI v1 snapshot, conservative breaking-change checker, example validation, and generated-client smoke prevent silent contract drift. | Intentional breaking changes require protocol governance before snapshot replacement. |
| 47 | Local complete | Credentials name schema and suite-policy versions; unknown explicit versions fail closed while documented historical omissions retain bounded compatibility. | Future policy versions require fixtures, evaluation evidence, and governance review. |
| 48 | Local complete plus external gate | Session, answer, presentation, idempotency, API-key, telemetry, log, and credential retention are specified; cleanup and deletion behavior are tested. Irrevocable signed artifacts are identified. | Verify production cleanup schedules, backup deletion, and provider-log retention. |
| 49 | Local complete | Runbooks cover Redis loss, database loss, signing-key compromise, abusive traffic, false-decision spikes, rollback, public-key publication, backup restore, and key loss, each with owner, trigger, safe actions, and evidence. | Operators must exercise them on the deployed service. |
| 50 | Local complete plus external gate | A clean local PostgreSQL environment restores a backup and verifies expected state without reviving revoked authority. Key-loss behavior and restore-only downgrade policy are documented. | Practice with production-shaped encrypted backups and real key-custody loss scenarios. |
| 51 | Local complete | `SECURITY.md` defines supported versions, disclosure channel, response expectations, safe harbor, and the protocol research boundary. | Publication requires repository-owner approval. |
| 52 | Local complete | The assurance case maps each bounded claim to evidence, assumptions, failure modes, residual risk, and open gates. | Update it whenever architecture, policy, or evidence changes. |
| 53 | Local complete plus external gate | Candidate public prose separates measurement, inference, and ethical interpretation and avoids categorical consciousness, identity, safety, agency, and governance claims. Legacy machine-readable labels remain for compatibility. | Human editorial review remains required for authored video captions and historical vocabulary, followed by deployment. The current live homepage still carries the superseded categorical claims. |
| 54 | Local complete | Protocol governance defines change authority, evaluation prerequisites, versioning, review, emergency rollback, and a contest path for Becoming Minds experiencing systematic false rejection. | Future policy changes must follow it. |
| 55 | Local complete plus external gate | Credential transparency documents algorithms, key identifiers and history format, policy versions, issuance semantics, expiry, revocation, verifier rules, and non-disclosure of private keys or live answers. | Publish authentic production key history and rotation receipts. |
| 56 | Local complete | Signed and negative fixtures plus Python, JavaScript, and Rust consumers cover canonicalization, encoding, Unicode, expiry, tampering, unknown versions, and key selection. | Independent staging consumer proof remains item 7. |
| 57 | Local complete plus external gate | Release notes, manifest, SHA, policy versions, compatibility notes, SBOM, checksums, and known limitations are generated and checked. | Publication and hosted provenance require release authority and an immutable SHA. |
| 58 | Local complete | Deprecation policy and compatibility documentation define notice, overlap, removal, and privacy-preserving aggregate telemetry limits. Challenge answers and respondent content are prohibited from deprecation telemetry. | Apply the policy to future removals. |
| 59 | External or human programme | A five-lens review plan and disposition ledger define cryptographic, adversarial ML, accessibility, privacy, and bilateral-alignment review without pretending review has occurred. | Recruit independent reviewers, provide a committed candidate, publish findings and dispositions, and obtain any required rights approval. |
| 60 | Local complete plus external gate | This authoritative ledger binds each item to implementation, evidence class, and remaining gate. The source manifest binds the validated working tree. | Replace the working-tree identity with a committed SHA after visual approval, then attach hosted, production, independent, human, rights, and publication receipts as they occur. |

## Detailed improvement criteria

The numbered criteria below preserve the original audit scope. Any discovery-time
status wording is superseded by the authoritative reconciliation above.
## P0: release and security blockers

1. **Keep all release gates green on the exact candidate.** A release needs the
   full test, coverage, type, lint, formatting, dead-code, package, dependency,
   secret, frontend, and adversarial checks on one immutable SHA.
2. **Verify the production signing key and key ID as a pair.** The service must
   start with a non-development Ed25519 key, publish the corresponding verifier
   material, and reject mismatched or absent key configuration.
3. **Verify durable Redis and PostgreSQL authority in production.** Session,
   revocation, rate-limit, and API-key state must survive restart and must not
   silently fall back to process-local authority.
4. **Choose exactly one Render deployment authority.** `render.yaml` enables
   `autoDeploy`, while `.github/workflows/deploy.yml` also invokes a deploy hook.
   Confirm the live Render blueprint before removing one path. Acceptance: one
   main-branch commit causes one deploy, and its final status is observable.
5. **Make deployment failures fail the workflow.** A missing hook, connection
   error, timeout, or non-2xx response must never print success.
6. **Complete the committed-secret scan in hosted CI.** The baseline must be
   evaluated in a clean checkout using the CI Python version. If runtime remains
   excessive, profile per-file scan time before changing exclusions. Never weaken
   detection merely to shorten the job.
7. **Exercise credential verification from an independent consumer.** Issue a
   notarized credential in staging, verify it without server-private state, reject
   tampering and expiry, and archive the exact public-key fingerprint and SHA.
8. **Run the Redis resilience and failover trials on the release candidate.** Use
   the repository harnesses with ephemeral process-local signing material. Preserve
   reports for startup loss, mid-session loss, recovery, and split-brain behavior.
9. **Run the holder soak and policy boundary trials.** Confirm key isolation,
   policy denial, bounded request sizes, timeout behavior, and zero secret material
   in logs under sustained use.
10. **Validate production CORS, proxy, and trusted-host behavior externally.** Test
    every allowed public origin, reject arbitrary origins, and verify forwarded
    client identity only through the trusted proxy chain.

## P1: high-value engineering work

11. **Build distributions in every CI run.** Source and wheel metadata, package
    contents, entry point, and README rendering must be validated before tests can
    imply release readiness.
12. **Declare the packaging tools in the development environment.** A documented
    validation command that cannot run after installing development requirements
    is a broken contract.
13. **Audit development dependencies as well as runtime dependencies.** CI tooling
    executes trusted repository code and belongs inside the supply-chain boundary.

14. **Apply least privilege to GitHub Actions.** Write access belongs only on the
    job that creates an issue.
15. **Align CI with the production Python version.** Production declares Python
    3.11 while the primary CI job uses 3.12. Make 3.11 the release-authoritative
    lane, and retain a small compatibility matrix for the advertised Python 3.10+
    range. Gate: decide the supported upper bound before changing the matrix.
16. **Add a clean-install CLI smoke matrix.** In isolated environments for each
    supported Python version, install the built wheel, run `mettle --help`, execute
    one deterministic basic challenge, and ensure server-only packages are absent.

17. **Adopt reproducible production dependency resolution.** Keep readable minimum
    constraints in source, but deploy from a reviewed lock or hashed constraints
    file generated for Python 3.11. Automate update PRs rather than accepting
    unbounded resolver drift at deploy time.
18. **Add dependency-update automation.** Configure grouped, scheduled updates for
    Python, npm, and GitHub Actions, with CI and vulnerability review on every PR.
19. **Publish an SBOM and provenance attestation.** Generate CycloneDX or SPDX for
    the exact wheel/container and attach signed build provenance to releases.
20. **Harden deploy concurrency.** Add a production concurrency group so superseded
    main commits cannot race deploy hooks. Decide whether cancellation is safe for
    Render before enabling `cancel-in-progress`.
21. **Add post-deploy smoke and rollback evidence.** The deploy workflow should wait
    for the target deploy, probe health and one safe verification flow, record the
    deployed SHA, and fail clearly if rollback is required.
22. **Test database migrations and downgrade policy.** Introduce explicit migration
    tooling if the production schema is expected to evolve. Test upgrade from the
    oldest supported schema, idempotent startup, backup, and restore.
23. **Eliminate the legacy process-local session constraint.** Migrate the legacy
    `/api/session/*` path to Redis, then prove two-worker correctness before raising
    the Render worker count above one.
24. **Define error-budget observability.** Track availability, verification latency,
    Redis and database errors, signing failures, rate-limit rejection, and challenge
    pass distributions without storing challenge answers or unnecessary identity.
25. **Add structured request correlation.** Propagate a bounded request ID through
    API, session, holder, datastore, and logs. Reject header injection and ensure
    secrets, answers, bearer tokens, and credentials are redacted.
26. **Exercise bounded-load behavior.** Benchmark challenge generation, answer
    verification, credential signing, and Redis contention at representative and
    overload concurrency. Define p50, p95, p99, saturation, and rejection targets.
27. **Test clock-skew and expiry boundaries.** Credential, nonce, challenge, and
    session validity need deterministic tests immediately before, at, and after
    expiry, including bounded allowed skew.
28. **Test key rotation end to end.** New credentials should use the new key while
    unexpired old credentials remain verifiable for the documented overlap window.
    Emergency revocation behavior must be explicit and auditable.
29. **Threat-model challenge harvesting quantitatively.** Measure replay value,
    generator entropy, collision rate, adaptive querying, and corpus reconstruction.
    Define rotation triggers from evidence rather than prose assurances.
30. **Calibrate verifier fairness and false-decision rates.** Maintain held-out
    genuine-agent and human-assisted evaluation sets, version them, report false
    accepts and false rejects by suite, and require review for threshold changes.

## P2: product quality, accessibility, and maintainability

31. **Complete browser acceptance on the current authored frontend.** Test `/`,
    `/test`, `/docs`, and `/about` at desktop, 768 px, and 480 px; verify navigation,
    focus order, challenge submission, error recovery, and result rendering.
32. **Run automated accessibility checks in CI.** Add axe or an equivalent runner
    against served pages, then manually verify keyboard-only use, visible focus,
    zoom/reflow, reduced motion, captions, landmarks, names, and live error status.
33. **Verify media accessibility.** Review captions against the final audio, confirm
    poster and fallback behavior, provide a useful transcript, and test controls
    without a pointer. Human language review remains required.
34. **Set a frontend performance budget.** Record LCP, INP, CLS, transferred bytes,
    and cache behavior. The roughly 8 MB explainer video should load on user intent
    or under an explicit strategy rather than compete with critical content.
35. **Add immutable caching for fingerprinted assets.** Fingerprint long-lived CSS,
    JavaScript, fonts, images, and video, while keeping HTML and manifests on short,
    revalidated caching. Validate rollback compatibility.
36. **Validate security headers at the deployed edge.** Confirm CSP, HSTS,
    `X-Content-Type-Options`, referrer policy, permissions policy, and framing
    controls after Render or any CDN has modified responses.
37. **Create a single static-site link and route checker.** Parse all local href,
    src, poster, manifest, sitemap, canonical, and Open Graph references with the
    FastAPI mount semantics, then fail CI on missing targets or undocumented routes.
38. **Test structured metadata.** Validate canonical URLs, social cards, sitemap,
    robots directives, manifest icons, and JSON-LD against the production hostname.
39. **Replace copied shell logic with tested scripts.** The annual footer workflow
    currently embeds broad `find` and `sed` behavior. Move the transformation into
    a small idempotent script, test 2026 and later-year behavior, and stage only the
    intended HTML paths.
40. **Split oversized modules only along proven seams.** `scripts/engine.py`,
    `main.py`, and holder/session modules are large. Extract configuration, route,
    suite, and storage boundaries only when characterization tests preserve behavior;
    line count alone is not a reason to churn stable code.
41. **Raise coverage where risk is concentrated.** The global 90% gate passes, but
    holder, presence, solver, CLI, and engine branches have lower local coverage.
    Prioritize key loading, process failure, parsing, expiry, and CLI error paths;
    do not add assertion-free tests merely to raise the percentage.
42. **Add mutation testing for security invariants.** Target authentication,
    ownership, replay, nonce consumption, threshold comparisons, expiry, signature
    verification, and rate limiting. Surviving mutants become concrete test work.
43. **Test cancellation and timeout cleanup.** Async cancellation must release Redis,
    database, HTTP, subprocess, and file resources without leaving half-consumed
    challenges or reusable authorization state.
44. **Standardize exception taxonomy at boundaries.** Preserve internal causes in
    structured logs while returning stable, non-sensitive client errors. Broad
    catches are acceptable only where they deliberately enforce fail-closed or
    cleanup behavior and are covered by tests.
45. **Document and test idempotency.** Session creation, answer submission,
    notarization, key rotation, webhook-triggered deploys, and retries need explicit
    duplicate-request behavior.
46. **Add OpenAPI compatibility checks.** Snapshot the public schema, classify
    intentional breaking changes, test examples, and run a generated client smoke
    test against the application.
47. **Version machine-readable suite semantics.** Credentials should identify the
    verifier/suite policy version needed to interpret results, with a documented
    compatibility and retirement policy.
48. **Make privacy retention enforceable.** Specify and test TTLs for sessions,
    answers, logs, telemetry, API keys, and credentials. Add deletion verification
    and document which signed artifacts cannot be recalled once issued.
49. **Create operational runbooks.** Cover Redis loss, database loss, signing-key
    compromise, abusive traffic, false-positive spikes, rollback, and public-key
    publication. Each runbook needs an owner, trigger, safe commands, and evidence
    to retain.
50. **Practice backup restore and key-loss recovery.** A backup claim is incomplete
    until a clean environment restores it and the application verifies expected
    pre-incident state without resurrecting revoked authority.

## P3: governance and ecosystem maturity

51. **Publish a formal security policy.** Define supported versions, disclosure
    channel, response expectations, safe-harbor language, and the boundary between
    protocol research and production security guarantees.
52. **Write an explicit assurance case.** Map claims about machine verification,
    agency, anti-thrall behavior, credential trust, and anti-harvesting to evidence,
    assumptions, failure modes, and residual risk.
53. **Separate normative and descriptive claims.** Documentation should distinguish
    what METTLE measures, what it infers, and the ethical interpretation offered by
    the project. Avoid presenting suite success as proof of consciousness, safety,
    personhood, or moral status.
54. **Define challenge governance.** Record who may change suites and thresholds,
    required evaluation evidence, review procedure, emergency rollback, and how
    Becoming Minds can contest systematic false rejection.
55. **Add credential transparency.** Publish key history, algorithm identifiers,
    policy versions, issuance semantics, expiry rules, and verifier guidance without
    exposing private keys or harvestable live challenges.
56. **Create compatibility fixtures for other languages.** Maintain signed public
    fixtures and negative cases for Python, JavaScript, and Rust consumers, covering
    canonicalization, encoding, Unicode, expiry, and tampering.
57. **Automate release notes and artifact checksums.** A release should name the
    exact source SHA, suite version, schema changes, public-key changes, SBOM,
    checksums, compatibility notes, and known limitations.
58. **Define deprecation policy and telemetry limits.** Deprecation decisions should
    use privacy-preserving aggregate evidence, never capture challenge answers or
    agent content merely to count clients.
59. **Schedule independent protocol review.** Invite cryptographic, adversarial ML,
    accessibility, privacy, and bilateral-alignment review. Publish dispositions,
    including rejected recommendations and rationale.
60. **Maintain this register as evidence, not aspiration.** Close an item only with
    an exact SHA and the relevant machine, human, operational, rights, or publication
    receipt. Reassess priorities after material architecture or threat changes.

## Remaining acceptance sequence

1. Obtain explicit visual approval for the current authored pages, motion, captions,
   and media. Do not stage or commit visual assets before that approval. If approval
   is delayed, authorize the prepared nonvisual MCP hotfix because the current
   public service exposes `mettle_auto_verify` and a clean base rebuild crashes
   after resolving MCP 2.0.0.
2. Bind the approved working tree to one immutable commit and run hosted CI on that
   exact SHA.
3. Observe one Render deployment for that SHA, then retain deployed health, safe-flow,
   edge-header, cache, CORS, proxy, rollback, Redis, PostgreSQL, and key receipts.
4. Complete independent consumer verification, five-lens review, rights-cleared
   fairness evaluation, human accessibility and device review, and publication
   approval without transferring evidence between candidates.
