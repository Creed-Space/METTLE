# METTLE comprehensive improvement register

Date: 2026-08-12

Candidate inspected: `86c515fb77f961ff33a5bbb11d058a8cc6504c01`

This register covers product correctness, security, resilience, privacy, API design,
operations, supply chain, testing, packaging, performance, accessibility,
documentation, governance, and maintainability. Items are separated by evidence
and gate so that a clean static check is never mistaken for runtime or production
acceptance.

## Evidence baseline

* `pytest`: 1,765 passed, total coverage 90.03%.
* Ruff lint and format: clean across 94 Python files.
* mypy with untyped bodies checked: clean across 94 Python files.
* Vulture at 80% confidence: no findings.
* Bandit: no findings in the production and red-team Python surfaces.
* `pip-audit -r requirements.txt`: no known vulnerabilities.
* `npm audit --audit-level=low`: no known vulnerabilities.
* Static JavaScript syntax and Impeccable design checks: clean.
* Local Python 3.14 dependency consistency: clean.
* The committed secret scan was started against the same 167-file scope as CI,
  but did not complete after several minutes on Python 3.14 and was interrupted.
  Hosted Python 3.12 CI remains the authoritative gate for this scanner.

## P0: release and security blockers

1. **Keep all release gates green on the exact candidate.** A release needs the
   full test, coverage, type, lint, formatting, dead-code, package, dependency,
   secret, frontend, and adversarial checks on one immutable SHA. Status: local
   gates are green except the incomplete secret scan; hosted CI is still required.
2. **Verify the production signing key and key ID as a pair.** The service must
   start with a non-development Ed25519 key, publish the corresponding verifier
   material, and reject mismatched or absent key configuration. Status: code and
   tests exist; production secret material cannot be inspected from this checkout.
3. **Verify durable Redis and PostgreSQL authority in production.** Session,
   revocation, rate-limit, and API-key state must survive restart and must not
   silently fall back to process-local authority. Status: extensive local tests
   exist; a production restart/failover receipt remains an operational gate.
4. **Choose exactly one Render deployment authority.** `render.yaml` enables
   `autoDeploy`, while `.github/workflows/deploy.yml` also invokes a deploy hook.
   Confirm the live Render blueprint before removing one path. Acceptance: one
   main-branch commit causes one deploy, and its final status is observable.
5. **Make deployment failures fail the workflow.** A missing hook, connection
   error, timeout, or non-2xx response must never print success. Status: implemented
   in this change with required-secret validation and fail-fast curl options.
6. **Complete the committed-secret scan in hosted CI.** The baseline must be
   evaluated in a clean checkout using the CI Python version. If runtime remains
   excessive, profile per-file scan time before changing exclusions. Never weaken
   detection merely to shorten the job. Status: hosted gate outstanding.
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
    imply release readiness. Status: implemented with `python -m build` and
    `twine check`.
12. **Declare the packaging tools in the development environment.** A documented
    validation command that cannot run after installing development requirements
    is a broken contract. Status: implemented with pinned `build` and `twine`.
    The first build also exposed deprecated setuptools license metadata; it was
    migrated to an SPDX expression and the redundant license classifier removed.
13. **Audit development dependencies as well as runtime dependencies.** CI tooling
    executes trusted repository code and belongs inside the supply-chain boundary.
    Status: implemented with a second `pip-audit` gate.
14. **Apply least privilege to GitHub Actions.** Write access belongs only on the
    job that creates an issue. Status: implemented for Red Council; unused
    `security-events: write` was removed.
15. **Align CI with the production Python version.** Production declares Python
    3.11 while the primary CI job uses 3.12. Make 3.11 the release-authoritative
    lane, and retain a small compatibility matrix for the advertised Python 3.10+
    range. Gate: decide the supported upper bound before changing the matrix.
16. **Add a clean-install CLI smoke matrix.** In isolated environments for each
    supported Python version, install the built wheel, run `mettle --help`, execute
    one deterministic basic challenge, and ensure server-only packages are absent.
    Status: the clean-wheel install and `--help` smoke are implemented in the main
    CI lane; the supported-version matrix and deterministic challenge remain.
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

## Recommended sequence

1. Merge the packaging, dependency-audit, deploy-failure, and workflow-permission
   fixes after clean hosted CI.
2. Resolve the single-versus-double Render deployment authority and add a post-deploy
   SHA-bound smoke check.
3. Run the release-candidate secret, Redis, holder, key-rotation, and independent
   credential-verification gates.
4. Close browser accessibility and performance acceptance on the current authored
   frontend before further visual redesign.
5. Implement reproducible dependency resolution, supported-Python smoke matrices,
   observability, migration, and multi-worker session work as bounded follow-ups.
6. Advance evaluation governance, assurance, interoperability, and independent
   review as an ongoing protocol programme.
