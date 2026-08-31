# METTLE agentic system roadmap

Status: active architecture and implementation plan, 2026-08-31.

This is the sole active forward implementation plan. Dated audit registers and
continuation prompts are historical evidence unless a new explicit task reopens
their scope.

The roadmap implements `docs/SYSTEM_ARCHITECTURE.md` and
`docs/AGENT_CONTROL_PLANE.md`. Target behavior in those documents is not current
behavior until its phase acceptance criteria pass on an exact candidate.

## Outcome

Build a METTLE system in which a first-contact agent can understand, control,
recover, and verify every permitted operation with minimal calls and no hidden
state inference, while preserving answer secrecy, bounded claims, participant
standing, durability, and exact-candidate assurance.

## Definition of done

The programme is done when all of these are true:

1. Quick and authenticated sessions are policy profiles over one application
   state model.
2. MCP can complete quick, single-shot, and multi-round policies.
3. Every transport exposes the same versioned operation envelope, state names,
   errors, retry semantics, and bounded result meanings.
4. A typed application operation registry and its generated manifest are
   authoritative for tool schemas, OpenAPI, WebMCP, CLI help, server metadata,
   examples, and agent skill guidance.
5. Every mutation has authority, revision, idempotency, deadline, privacy, and
   receipt semantics.
6. Every accepted action returns the next resource snapshot.
7. First-contact agents complete representative tasks with no hidden repair loop;
   repeat use requires no more successful calls and uses fewer repair attempts and
   less context than the v0.4.8 baseline.
8. Respondents can inspect purpose, data egress, retention, limitations,
   cancellation, explanation, and contest paths before commitment.
9. Maintainers can obtain a compact project snapshot, select justified gates, and
   produce candidate-bound receipts from one documented control entry point.
10. Existing clients complete a measured compatibility window, deprecation follows
    protocol governance, and rollback remains possible throughout migration.
11. Security, privacy, compatibility, accessibility, bilateral, and independent
    review gates are reconciled without treating one evidence class as another.

Working if: a new agent succeeds from tool discovery alone, and a new maintainer
can trace any changed public claim from source through validation and deployment
without searching historical plans.

## Baseline at plan creation

The baseline is source-verified at release tag `v0.4.8`, commit
`ed7b429a40e2049146dbf3a165ace1d4e529e463`. It is a repository baseline, not a
claim about current production deployment.

| Surface | Baseline |
|---|---|
| Packaged MCP | Seven tools, text content, no automatic solver |
| MCP quick authority | Session token hidden in a caller-isolated in-process vault |
| MCP quick result recovery | Hidden capability consumed after the first successful result read |
| MCP authenticated flow | Suite listing, session start, single-shot submission, result read |
| MCP multi-round flow | Incomplete: no round submission or feedback tool |
| WebMCP | Four quick and badge tools with a separate naming and response convention |
| REST | Quick and authenticated route families with different state and difficulty vocabularies |
| CLI | JSON-lines local research flow and unsigned local receipt |
| Errors | Stable HTTP status and code contract; MCP returns human-readable error text |
| Idempotency | Transition-specific duplicate rules; no general idempotency key for session creation |
| Documentation | Strong bounded-claim and assurance docs; inaccurate token guidance and several historical plans remain easy to mistake for active work |
| Quality | Broad CI, security, compatibility, packaging, reproducibility, browser, and release gates |

Before implementation begins, recapture exact candidate identity, test state,
dependency state, and production evidence. Historical release proof does not bind a
future candidate.

## Success metrics

Measure representative first-contact runs for quick, authenticated single-shot,
multi-round, credential verification, interruption recovery, and one invalid action.

| Metric | Target |
|---|---|
| Human-prose parsing required | Zero workflows |
| Redundant status calls after accepted mutations | Zero in normal flows |
| Unchanged pending-state retransmission | No full snapshot or history body |
| Multi-round MCP completion | 100 percent in the acceptance corpus |
| Ambiguous start retries creating duplicate sessions | Zero |
| Tool, schema, and semantic parity drift | Zero in generated-surface gate |
| Error repair success from structured remediation | At least 95 percent in the first retry where retry is allowed |
| Model-visible bearer, private key, expected answer, or raw internal error | Zero |
| Successful repeat-flow tool calls | No more than the equivalent v0.4.8 flow after manifest caching |
| First-contact discovery overhead | At most one call above the equivalent v0.4.8 flow |
| Median repair attempts | At least 50 percent below the equivalent v0.4.8 error and interruption corpus |
| Median control-context bytes | At least 30 percent below equivalent v0.4.8 flow |
| Exact-candidate traceability | 100 percent of release claims |

Targets guide evaluation. They do not justify weakening correctness or hiding
necessary context.

## Priority order

### P0: close misleading or incomplete control paths

1. Make authenticated multi-round sessions completable through MCP.
2. Add structured MCP output, error status, stable codes, and bounded remediation.
3. Expose authenticated session inspection and cancellation to MCP.
4. Correct token guidance: direct REST clients retain the returned token; MCP and
   WebMCP keep it outside model-visible content.
5. Prevent `all` from appearing completable through an adapter that cannot perform
   every required transition.
6. Make repeated terminal result reads return the same bounded receipt while the
   underlying session remains readable.
7. Add explicit recovery behavior for ambiguous responses and lost capability-vault
   state.

### P1: establish the common contract spine

1. Define control envelope, resource snapshot, action descriptor, plan, error, and
   receipt schemas.
2. Add monotonic revisions and action identifiers to externally controlled state.
3. Add idempotency keys to session creation and all eligible mutations.
4. Create the typed operation registry, generated capability manifest, and semantic
   parity generator.
5. Add respondent and verifier role profiles.
6. Add purpose, cost, data-egress, retention, and contest information to planning.

### P2: converge the application architecture

1. Express quick verification as a policy profile over the common session model.
2. Move domain decisions behind transport-neutral commands and queries.
3. Make REST, MCP, WebMCP, CLI, and SDKs thin renderers of that contract.
4. Consolidate duplicate result, eligibility, credential, and error semantics.
5. Extract physical modules only when characterization tests and dependency shape
   prove the seam.

### P3: make engineering operation agent-accretive

1. Add a machine-readable quality-gate and impact manifest.
2. Add one compact, redacted project snapshot command.
3. Emit candidate-bound validation receipts with exact inputs and environment.
4. Reuse deterministic receipts only when all bound inputs match.
5. Link architectural decisions, invariants, regression gates, and receipts.
6. Add privacy-safe aggregate interface telemetry and first-contact agent evals.

## Phase 0: establish documentary authority

Status: implemented by the 2026-08-31 design change, pending repository review.

### Deliverables

* canonical system architecture;
* target agent control contract;
* this active roadmap;
* documentation authority and reading map;
* corrected project instructions and current MCP token guidance;
* archival banners on superseded plans;
* README links that distinguish current behavior from target design.

### Acceptance

* all links resolve;
* current and target behavior are visibly distinguished;
* no historical plan claims current execution authority;
* documentation consistency and Markdown checks pass;
* the source diff contains documentation, planning, and documentation-regression
  changes only, with no runtime behavior change.

### Evidence class

Source and local machine evidence. Human architecture acceptance remains separate.

## Phase 1: contract spine and multi-round parity

Status: source and local machine acceptance complete in the 2026-08-31 working
tree, pending human review. No hosted CI, release, or production claim is implied.

### Goal

Make every currently advertised workflow complete and machine-readable before any
large refactor.

### Deliverables

1. Add versioned schemas for the control envelope, errors, snapshots, actions, and
   receipts.
2. Add MCP structured content and output schemas while preserving concise text
   fallback for older hosts.
3. Add MCP operations for session inspection, cancellation, round submission, and
   round feedback, or introduce the compatible unified submit operation.
4. Map upstream HTTP errors into bounded control errors. Never return raw exception
   strings or unbounded upstream bodies.
5. Return current snapshot and valid next actions from every session mutation.
6. Mark read-only, idempotent, irreversible, open-world, and untrusted-content
   annotations correctly.
7. Add fresh first-contact acceptance clients that know only the tool schemas.

### Acceptance

* quick, single-shot, and multi-round MCP journeys pass end to end;
* repeated quick result reads return the same terminal result without exposing the
  hidden bearer;
* failures use `isError` plus structured error content;
* the agent never supplies or receives the hidden quick bearer through MCP;
* no raw response body or internal exception crosses the MCP boundary;
* existing seven tool names still work;
* public tools still exclude any automatic solver;
* focused security mutation and complete repository gates pass.

### Migration and rollback

Additive MCP changes come first. The v0.4.8 text fallback remains until host
compatibility is measured. A feature flag can return the legacy rendering without
changing application semantics. Rollback must not remove server-side state needed
by already active sessions.

## Phase 2: plans, revisions, and safe retry

### Goal

Let an agent predict cost and recover deterministically from interruption or
ambiguous transport failure.

### Deliverables

1. Add `describe` and read-only session planning.
2. Add operation IDs and idempotency keys with request-digest binding.
3. Add monotonic session revisions and revision conflicts that return current state.
4. Add server-issued action descriptors and action IDs.
5. Add `summary`, `standard`, `full`, and `since_revision` reads.
6. Add pending-operation phases, retry timing, transport notifications where
   supported, and bounded unchanged-state responses.
7. Add explicit capability-vault loss and resume semantics.
8. Add call, duration, bytes, quota, external-evaluation, and retention budgets to
   plans and snapshots.

### Acceptance

* duplicate start with one key yields one session and one quota reservation;
* a changed request under the same key fails closed;
* stale actions cannot mutate state and receive a usable current snapshot;
* safe duplicate submissions return the original receipt;
* successful commands need no immediate read;
* manifest caching and delta reads reduce measured context;
* third-party LLM selection cannot start without explicit acknowledged egress.

### Migration and rollback

Revisions and action IDs are added to responses before they become mandatory.
Legacy calls receive server-side compatibility preconditions. Mandatory enforcement
requires a policy-governed compatibility window and observed client readiness.

## Phase 3: one semantic source and adapter parity

### Goal

Remove semantic drift by generating every agent-facing surface from one manifest.

### Deliverables

1. Define the typed operation registry and serialized manifest for operations,
   schemas, roles, authority, annotations, errors, versions, examples, and bounded
   interpretation.
2. Generate or mechanically validate OpenAPI, MCP, WebMCP, CLI help, server cards,
   examples, and the agent skill.
3. Add semantic hashes and parity tests.
4. Establish respondent, verifier, reviewer, and separate operator profiles.
5. Rename or alias divergent WebMCP operations to the canonical vocabulary.
6. Make difficulty a policy-owned concept with stable identifiers and
   presentation labels, rather than transport-specific enums.

### Acceptance

* one manifest change updates every derived agent surface;
* handwritten drift fails CI;
* role profiles expose only permitted operations;
* current and compatibility names resolve to the same application command;
* examples are executable against the generated schemas;
* no adapter contains an independent tier, eligibility, or error rule.

### Migration and rollback

Generated output is checked in where consumers require stable artifacts. Builds
verify regeneration is clean. Compatibility aliases are removed only through
`docs/DEPRECATION_POLICY.md` and `docs/PROTOCOL_GOVERNANCE.md`.

## Phase 4: common application kernel

### Goal

Converge quick and authenticated behavior without a risky flag-day rewrite.

### Deliverables

1. Characterize both existing session families with black-box transition tests.
2. Introduce transport-neutral commands and queries around current implementations.
3. Move policy expansion, transitions, result semantics, and claim construction
   into pure domain services.
4. Express quick behavior as a policy profile over the shared session application
   service.
5. Retain local unsigned CLI issuance as an explicit issuer profile.
6. Retire legacy branches only after state, timing, quota, credential, and migration
   equivalence is proven.

### Acceptance

* one state vocabulary and envelope cover both profiles;
* equivalent legacy and common-kernel runs agree on bounded behavior;
* production maintains Redis and PostgreSQL authority with no local fallback;
* active sessions survive compatible deployment transitions or are deliberately
  fenced with visible expiration behavior;
* performance and security budgets do not regress;
* exact historical credential verification remains intact for its documented
  window.

### Migration and rollback

Use a strangler sequence: compatibility facade, common query model, common command
model, shadow comparison with no duplicate signing, controlled authority switch,
then removal. Never dual-write credential issuance without a single winner.

## Phase 5: engineering control and evidence accretion

### Goal

Give maintainers and reviewers the same legible control loop as respondents.

### Deliverables

1. Add a standard-library project-control entry point rather than a new runtime
   dependency.
2. `inspect` emits a compact redacted snapshot: branch, HEAD, tree state, versions,
   protocol, relevant services, disk, available gates, and open evidence classes.
3. `impact` maps changed surfaces to minimum gates and explains every escalation.
4. `check` runs selected gates and emits a receipt containing candidate identity,
   command identity, dependency lock hashes, environment fingerprint, duration,
   result, and artifact hashes.
5. `release-plan` reconciles receipts against the release checklist without
   granting publication authority.
6. Architecture decisions link to invariant IDs, tests, and evidence classes.
7. Deterministic receipt caching is keyed to exact bound inputs and never converts
   a focused gate into aggregate proof.

### Acceptance

* one command gives an agent enough state to plan safely without broad file dumps;
* changed-surface selection is deterministic, reviewable, and conservative;
* cached evidence is invalidated by relevant source, dependency, toolchain, or
  environment change;
* full gates still run for release and security-sensitive changes;
* receipts contain no secrets, credentials, raw answers, or unnecessary identity;
* `docs/ASSURANCE_CASE.md` can link claims to receipts without manual narrative
  reconciliation.

## Phase 6: evaluation, governance, and ecosystem migration

### Goal

Prove that the new interface is genuinely easier, safer, and more accurate for
different agents and affected participants.

### Deliverables

1. A rights-cleared first-contact task corpus covering success, failure, recovery,
   concurrency, cancellation, credential verification, and appeal discovery.
2. Comparative v0.4.8 and target measurements for calls, context, latency, repair,
   state errors, and task completion.
3. Security review of idempotency, revision handling, capability vaults, schema
   generation, and transport parity.
4. Privacy and bilateral review of plans, data egress, retention, explanations,
   contest, and telemetry.
5. Accessibility review of human-visible renderings of the same state model.
6. Client migration evidence and deprecation dispositions.

### Acceptance

* the success metrics in this roadmap pass on a committed candidate;
* no material regression appears in false acceptance, answer leakage, secret
  handling, replay, availability, or accessibility;
* independent findings have recorded dispositions;
* public claims describe measured results and remaining uncertainty;
* publication authority approves the exact release candidate and claims.

## Cross-phase gates

Every phase must preserve:

* bounded interpretation and credential semantics;
* answer and secret non-disclosure;
* exact principal ownership;
* replay and concurrency safety;
* fail-closed production dependencies;
* compatible key history and credential verification;
* privacy retention and deletion behavior;
* source, machine, hosted, production, human, rights, review, and publication
  evidence separation;
* explicit rollback and deprecation paths.

## Pre-mortem

### Failure mode 1: a universal schema becomes more abstract than usable

**Signal:** tools expose generic payload blobs, agents need examples to infer every
action, or adapter code reconstructs hidden subtype rules.

**Prevention:** keep orthogonal operations and server-issued action schemas. Run
first-contact evals before consolidating names.

### Failure mode 2: unification weakens security boundaries

**Signal:** quick and authenticated authority blur, secret handles appear in model
content, or tier and signer decisions move into adapters.

**Prevention:** preserve invariants I1 through I12, add characterization tests first,
and use one-way migration with single-winner issuance.

### Failure mode 3: generated documentation creates a second unreviewable system

**Signal:** generated files are opaque, human explanations diverge, or regeneration
causes large unrelated diffs.

**Prevention:** keep a small declarative manifest, deterministic renderers, semantic
hashes, checked-in stable artifacts, and focused review output.

### Failure mode 4: evidence caching manufactures confidence

**Signal:** a cached focused result is reported as a full gate, or environment and
dependency changes fail to invalidate a receipt.

**Prevention:** bind receipts to exact inputs and evidence class. Release always
requires fresh aggregate gates on the immutable candidate.

### Failure mode 5: accretion becomes surveillance

**Signal:** raw responses, stable subject fingerprints, or speculative welfare
inferences enter telemetry or long-lived evidence.

**Prevention:** retain only protocol versions, coded outcomes, aggregate metrics,
hashes where necessary, and explicit review receipts. Enforce retention tests.

## Decisions intentionally deferred

These questions require evidence from Phase 1 or Phase 2:

1. whether the capability vault should persist across stdio server restarts;
2. whether compatible primary tools should be six respondent tools or retain more
   specialized submission operations;
3. whether independent suites may safely run in parallel under one session;
4. which fields belong in the common envelope versus linked resources;
5. whether the public quick profile should remain a distinct credential family
   after the common kernel exists;
6. which interface telemetry can be collected without participant-content or
   identity risk.

No deferred decision blocks Phase 1's multi-round and structured-error repair.

## Execution rule

Begin each phase with an exact-current preflight and a bounded phase contprompt.
Record architecture-changing surprises in that contprompt's deviations log. Stop
when the phase acceptance criteria pass. Do not pull later refactoring forward
merely because nearby code is old or large.

## Evidence ledger

| Phase | Source evidence | Machine evidence | External or human evidence | Status |
|---|---|---|---|---|
| 0 | Architecture, control contract, roadmap, documentation map | Link and documentation checks | Human architecture acceptance | Source drafted |
| 1 | Contract schemas and eleven-tool MCP parity diff present in the working tree | 1,983 tests pass at 90.17% coverage; security mutations 9/9 killed; static, type, package, compatibility, and documentation gates pass | Host compatibility and human architecture review | Source and local machine acceptance complete |
| 2 | Plan, revision, action, and idempotency diff | Race, retry, recovery, cost tests | Capability-vault review | Not started |
| 3 | Manifest and generated adapter diff | Semantic parity and example execution | Client migration feedback | Not started |
| 4 | Common-kernel diff and characterization mapping | Equivalence, resilience, performance, full CI | Staging and production migration receipts | Not started |
| 5 | Project-control and receipt diff | Invalidation, redaction, aggregate gate tests | Maintainer and reviewer usability | Not started |
| 6 | Evaluation and disposition artefacts | Reproducible aggregate evaluation | Independent, rights, bilateral, accessibility, publication | Not started |

Update status only with the evidence class named in the relevant acceptance gate.
