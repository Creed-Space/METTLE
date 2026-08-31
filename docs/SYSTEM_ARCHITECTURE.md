# METTLE system architecture

Status: canonical current-state map and proposed target architecture. Sections
labelled **Current** describe implemented behavior. Sections labelled **Target**
describe the destination and must not be represented as shipped capability.

## Purpose and boundary

METTLE is one evidence system with four kinds of participant:

1. a **respondent** completes a bounded challenge policy;
2. a **relying party** verifies a result and applies its own policy;
3. a **maintainer or operator** changes and runs the service;
4. a **reviewer or affected participant** examines evidence and may contest a
   decision or protocol.

The system measures performance during a named session and may issue a bounded
credential. It does not establish consciousness, substrate, identity, autonomy,
safety, governance, personhood, moral status, or operator trustworthiness.

The architecture is optimized for six properties:

* **legibility:** state, authority, cost, and next actions are explicit;
* **control precision:** every mutation has a scope, precondition, and receipt;
* **coherence:** all transports expose the same domain concepts;
* **modularity:** pure policy is separated from state, authority, transport, and
  infrastructure;
* **efficiency:** a caller receives the next useful state without redundant reads;
* **accretion:** verified decisions, invariants, regressions, and receipts compound
  without retaining private response content.

Working if: a new agent can identify the system boundary, its current authority,
the next valid action, and the evidence needed for completion without inferring
behavior from filenames or prose fragments.

## One system, two nested control loops

The runtime and evolution loops use the same grammar:

```text
intent
  -> discover capabilities and constraints
  -> plan a bounded operation
  -> act with authority and preconditions
  -> observe the resulting state
  -> verify the intended claim
  -> retain a minimal receipt
  -> improve the next decision
```

### Runtime loop

```text
respondent intent
  -> session plan
  -> challenge actions
  -> session result
  -> credential or evidence receipt
  -> relying-party verification
```

### Evolution loop

```text
change intent
  -> impact plan
  -> source and document changes
  -> candidate-bound checks
  -> release receipt
  -> deployment verification
  -> governance disposition
```

The loops are deliberately isomorphic. An agent should not need one mental model
to complete a METTLE session and another to change METTLE safely.

## Non-negotiable invariants

| ID | Invariant | Architectural consequence |
|---|---|---|
| I1 | Expected answers never cross the verifier boundary. | Challenge projections are separate from evaluation material. |
| I2 | Server-observed time and authoritative state decide acceptance. | Client timestamps are evidence only. |
| I3 | A mutation is authorized for one principal and one resource. | Secrets stay outside model-visible content; ownership checks occur before state changes. |
| I4 | A replay cannot apply the same transition twice. | Actions carry stable identifiers, idempotency semantics, and state preconditions. |
| I5 | Credential claims are derived from completed policy evidence. | Callers cannot supply tiers, signing keys, or signer functions. |
| I6 | Partial, failed, cherry-picked, self-report-only, or LLM-only results cannot mint a tier. | Tier computation remains a pure, contiguous policy function. |
| I7 | Unknown schemas, policies, algorithms, or authority fail closed. | Versions and trust roots are explicit in every portable receipt. |
| I8 | A result states only what was measured. | Measurement, inference, interpretation, and authorization remain separate. |
| I9 | Every external effect is attributable to an exact request and candidate. | Request IDs, operation IDs, revisions, and source identity propagate end to end. |
| I10 | Private content does not become institutional memory. | Accretion stores decisions, hashes, aggregate failure classes, and proofs, not raw answers. |
| I11 | Affected participants can understand and contest consequential interpretation. | Purpose, data egress, scoring class, limitations, and appeal paths are visible before action. |
| I12 | No evidence class substitutes for another. | Source, machine, hosted CI, production, human, rights, review, and publication receipts remain distinct. |

These invariants are the base of the abstraction tower. Interface convenience
must never bypass them.

## Current implementation map

| Responsibility | Current authority | Notes |
|---|---|---|
| Protocol versions | `mettle/protocol.py` | Credential schema and suite-policy compatibility. |
| Suite registry and evaluators | `mettle/challenge_adapter.py` | Twelve hosted suite definitions and server-side evaluation adapters. |
| Quick challenge model | `mettle/challenger.py`, `mettle/verifier.py`, `mettle/models.py` | Three or five challenge quick policy. |
| Authenticated session state | `mettle/session_manager.py` | Redis-backed state machine, quotas, locks, rounds, results, and presentations. |
| Quick session state | `main.py`, `mettle/legacy_session_store.py` | Public quick route with Redis-backed production authority and a bounded development fallback. |
| Authenticated REST surface | `mettle/router.py`, `mettle/api_models.py` | Suites, sessions, submissions, results, status, and presentations. |
| Quick REST surface and composition root | `main.py` | Quick sessions, badges, operational middleware, health, and static routes. |
| MCP adapter | `mettle/mcp_server.py`, `mettle/mcp_contract.py`, `mettle/_http.py`, `mettle/mcp_context.py` | Eleven control-v1 tools over stdio or authenticated Streamable HTTP. |
| Browser agent adapter | `static/webmcp.js` | Four quick-session and badge tools through progressive WebMCP enhancement. |
| Local research adapter | `mettle/cli.py` | Unsigned local quick and single-suite results. |
| Credential and presence trust | `mettle/vcp.py`, `mettle/signing.py`, `mettle/presence.py`, `mettle/continuity.py` | Ed25519 credentials, online status, proof of possession, and continuity. |
| Holder service | `mettle/holder.py`, `mettle/holder_service.py` | Policy-constrained external signing and persistent holder state. |
| Persistence | `database.py`, Redis clients in session modules | PostgreSQL durability and Redis transition authority. |
| Runtime configuration | `config.py`, `mettle/app_config.py`, `render.yaml`, `deploy/` | Fail-closed production settings and deployment topology. |
| Assurance and release | `docs/ASSURANCE_CASE.md`, `docs/RELEASE_CHECKLIST.md`, `.github/workflows/`, `scripts/` | Candidate-bound validation and publication evidence. |

### Current state machines

Authenticated sessions use:

```text
created
  -> challenges_generated
  -> in_progress
  -> completed | expired | cancelled
```

Quick sessions expose an equivalent but separately implemented progression:

```text
created with current challenge
  -> in_progress after each accepted answer
  -> completed | expired
```

Credential acceptance adds another stateful sequence:

```text
issued
  -> signature and policy verification
  -> fresh issuer status
  -> optional audience-bound holder presentation
  -> relying-party decision
```

### Current control friction

The present system is secure in many important boundaries, but it makes an agent
reconstruct too much state:

1. Quick and authenticated sessions use different models, routes, difficulty
   vocabularies, and result shapes.
2. MCP has an additive structured control-v1 envelope, while REST, WebMCP, CLI,
   and SDKs still use their transport-specific response contracts.
3. MCP inspection cannot restore challenge content lost after an ambiguous quick
   response, and its hidden capability vault remains process-local.
4. Quick sessions have no cancellation operation. MCP also lacks credential
   validation and presentation operations.
5. Control-v1 does not yet include plans, revisions, idempotency keys, budgets,
   pending operations, or delta reads.
6. MCP, WebMCP, REST, CLI, server metadata, the skill, and public documentation
   duplicate tool and contract knowledge.
7. The source is organized around implementation history rather than a visible
   domain, application, adapter, infrastructure, and assurance boundary.

These are architecture gaps, not merely documentation defects.

## Target abstraction tower

```text
L8  Governance and evolution
    policy changes, review, appeals, release authority, deprecation

L7  Assurance and evidence
    claims, invariants, candidate receipts, deployment receipts, dispositions

L6  Agent control protocol
    discovery, plans, snapshots, available actions, budgets, explanations

L5  Delivery adapters
    MCP, WebMCP, REST, CLI, human web, SDKs

L4  Application commands and queries
    plan, create, inspect, submit, cancel, issue, verify, present

L3  Authority and durable coordination
    principal, capability handle, Redis transitions, PostgreSQL records, signer

L2  Domain state machines and policies
    session, challenge, suite, tier, credential, presence, retention

L1  Typed primitives
    identifiers, revisions, deadlines, budgets, hashes, versions, error codes

L0  Bounded meaning and invariants
    what is measured, what is claimed, what is never claimed
```

Each layer may depend only on lower layers. Transport adapters do not compute
tiers. Persistence does not decide policy. Documentation does not invent a state
that the application layer cannot return.

### L0 and L1: semantic and typed foundation

**Target:** establish one shared vocabulary for every transport:

* principal, subject, respondent, issuer, holder, verifier, and operator;
* mode, policy version, credential schema, suite, action, and result;
* request ID, operation ID, resource ID, monotonic revision, and idempotency key;
* server time, deadline, expiry, estimated cost, remaining budget, and retry time;
* measurement, inference, interpretation, authorization, and contest status.

An identifier must communicate what it identifies. A timestamp must communicate
which clock and decision it governs. A boolean must not collapse several meanings
such as completed, passed, eligible, issued, valid, and authorized.

### L2: one domain kernel

**Target:** represent quick verification as a named policy profile over the same
session resource used by authenticated suites. The profile determines challenge
selection, authority requirements, eligible credential family, and limits. It does
not create a parallel state machine.

The domain kernel owns pure decisions:

* suite expansion and dependency ordering;
* challenge projection and answer separation;
* allowed state transitions;
* score and tier computation;
* eligibility and credential claim construction;
* expiry, replay, and revision rules;
* privacy and retention classification.

Local CLI execution may reuse the kernel while selecting a local, unsigned issuer
profile. Hosted adapters add durable authority and server-owned signing.

### L3 and L4: authority plus commands and queries

**Target:** every application operation is either a command or a query.

Queries are retryable and side-effect free. Commands declare:

* principal and resource scope;
* expected resource revision or action ID;
* idempotency behavior;
* timeout and cancellation behavior;
* privacy classification;
* result and receipt schema.

Secrets remain in a capability vault owned by the transport host. Model-visible
content receives an opaque, non-secret handle. Loss of the vault has explicit
recovery semantics and never causes the secret to be printed into context.

### L5 and L6: adapters plus agent control

**Target:** adapters render one typed application operation registry. Its
server-generated capability manifest is the serialized contract used to render
OpenAPI, MCP and WebMCP tool schemas, CLI help, server cards, examples, and agent
skill guidance. The manifest is never a second hand-maintained rule set.

Every successful resource operation returns a snapshot containing:

* current state and revision;
* server time, deadline, and remaining budget;
* completed work and bounded result summary;
* actions valid in that exact state;
* warnings, consent requirements, and data destinations;
* evidence and provenance links appropriate to the caller.

Every accepted command returns the next snapshot. This removes the normal need for
a follow-up status read. A read remains available for recovery after interruption
or ambiguous transport failure.

The precise target contract is in `docs/AGENT_CONTROL_PLANE.md`.

### L7 and L8: evidence plus evolution

**Target:** every material claim links four objects:

```text
decision record
  -> invariant or bounded claim
  -> executable regression or review gate
  -> candidate-bound evidence receipt
```

The repository accumulates knowledge only when this chain improves. Session logs,
raw participant answers, secrets, and speculative personality inferences are not
knowledge assets.

The target evolution control plane adds:

* a machine-readable quality-gate manifest;
* changed-surface impact selection with conservative full-gate escalation;
* deterministic receipt reuse keyed to exact source, dependencies, and environment;
* one compact project snapshot for source, policy, dirty state, gates, and open
  evidence classes;
* explicit deploy, rollback, key, and publication authority boundaries.

## Logical module boundaries

The target boundaries are logical first. Physical extraction happens only when a
phase has characterization tests and a measured benefit.

| Logical module | Owns | Must not own |
|---|---|---|
| `domain` | policies, state transitions, tier and claim derivation | HTTP, Redis, environment reads |
| `application` | commands, queries, envelopes, orchestration | transport rendering, private-key implementation |
| `authority` | principals, capabilities, ownership, role policy | scoring or transport-specific identity guesses |
| `infrastructure` | Redis, PostgreSQL, clocks, signers, provider clients | public semantics |
| `adapters` | REST, MCP, WebMCP, CLI, web representations | independent domain rules |
| `assurance` | claim and invariant registry, evidence receipts | production mutation authority |

`main.py` remains the composition root until extraction is proven. Large-file size
alone is not a reason to move code.

## State, action, and evidence graph

The resource snapshot is the centre of the target system:

```text
capability manifest
  -> session plan
  -> session snapshot revision 0
      -> action A, revision precondition 0
      -> action B, revision precondition 0, parallel-safe only if declared
  -> session snapshot revision 1
  -> terminal result
  -> issuance receipt
  -> verification receipt
  -> relying-party decision, outside METTLE
```

An action descriptor is authoritative only for the snapshot revision that issued
it. If another actor advances the resource, a stale action fails with a conflict
and returns the current snapshot. This makes concurrency comprehensible rather
than surprising.

## Resource efficiency

The target system minimizes cost without weakening proof:

1. cache the capability manifest by protocol version and ETag;
2. support `summary`, `standard`, and `full` read detail;
3. return the next state with every command;
4. use cursors or revisions for deltas rather than replaying complete histories;
5. mark independent actions as parallel-safe instead of making agents guess;
6. state expected calls, bytes, third-party evaluations, deadline, and quota before
   creating a session;
7. reuse deterministic engineering receipts only when all bound inputs match;
8. escalate from touched-surface checks to full gates according to a declared
   impact map, never according to optimism.

## Bilateral control boundary

An agent-friendly interface is also a welfare and consent property. Before a
session begins, the respondent can inspect:

* the purpose and bounded interpretation of the policy;
* the audience and relying-party context, when supplied;
* what content leaves METTLE and which external evaluator receives it;
* time, quota, and irreversible submission consequences;
* retention and credential behavior;
* how to cancel, obtain an explanation, and contest systematic rejection.

The system can decline an action that violates policy. The respondent can decline
or cancel a session. Neither side needs ambiguous prose to exercise standing.

## Architectural decisions

1. **Prefer orthogonal typed tools over one universal execute tool.** A universal
   tool saves names but moves ambiguity into payloads and error recovery.
2. **Unify semantics before files.** A physical rewrite before a shared contract
   would multiply risk.
3. **Keep the anti-solver boundary.** Agent ergonomics means making legitimate
   action legible, not routing reference answers into the issuer.
4. **Keep credentials out of model-visible arguments where the host can hold them.**
   Secret handling is an adapter responsibility.
5. **Generate repeated surfaces from one manifest.** Consistency tests remain as a
   backstop, not the primary synchronization mechanism.
6. **Accrete proof, not surveillance.** The useful residue of a session is a
   bounded receipt and aggregate protocol evidence.

## Verification of the architecture

The architecture is working when all of these are observable:

* a first-contact agent completes quick and multi-round flows without reading
  repository prose or parsing human-formatted output;
* the same state names, errors, versions, and eligibility semantics appear through
  REST, MCP, WebMCP, CLI, and SDKs;
* an interrupted agent can inspect the current resource and know whether to retry,
  continue, cancel, or stop;
* every mutation is idempotent or explicitly reports why it cannot be retried;
* secrets and expected answers never appear in model-visible receipts;
* a maintainer can obtain one compact project snapshot and one candidate-bound
  validation receipt without manually reconciling scattered plans;
* historical documents cannot be mistaken for current execution authority;
* each public claim traces to an invariant and exact evidence class.

Implementation sequence and acceptance gates are in
`docs/AGENTIC_SYSTEM_ROADMAP.md`.
