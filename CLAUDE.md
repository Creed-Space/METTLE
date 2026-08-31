# CLAUDE.md: METTLE

## Project boundary

METTLE is an experimental reverse CAPTCHA with twelve registered challenge
suites. It measures performance during a named policy session and may issue a
signed, time-limited credential. A result does not establish identity,
non-human substrate, consciousness, autonomy, safety, governance, personhood,
moral status, or operator trustworthiness.

Preserve the distinction between:

* measurement during one session;
* an inference supported by that measurement;
* an ethical interpretation;
* a relying party's independent authorization decision.

## Read first

1. `docs/DOCUMENTATION_MAP.md`: authority, reading order, and historical boundaries.
2. `docs/SYSTEM_ARCHITECTURE.md`: current system map, invariants, and proposed target.
3. `docs/AGENTIC_SYSTEM_ROADMAP.md`: sole active forward plan.
4. The source and tests for the surface being changed.

`docs/AGENT_CONTROL_PLANE.md` is target design, not implemented behavior. Dated
audits, remediation reports, and `_contprompts/` are historical unless Nell opens a
new bounded execution task.

## Current implementation map

```text
main.py                         FastAPI composition root, quick API, operational middleware, static routes
config.py                       main service settings and fail-closed production configuration
database.py                     PostgreSQL models, migrations, retention, and durable records

mettle/challenge_adapter.py     twelve-suite registry, generators, and evaluators
mettle/challenger.py            quick challenge generation
mettle/verifier.py              quick challenge verification and result computation
mettle/api_models.py            authenticated API request and response models
mettle/router.py                authenticated suite, session, result, status, and presentation routes
mettle/session_manager.py       Redis-backed authenticated session authority
mettle/legacy_session_store.py  Redis-backed quick-session authority
mettle/mcp_server.py            packaged eleven-tool MCP adapter
mettle/mcp_contract.py          MCP control-v1 schemas, snapshots, actions, receipts, and errors
mettle/_http.py                 bounded authenticated Streamable HTTP MCP transport
mettle/cli.py                   unsigned local research CLI
mettle/vcp.py                   credential construction, verification, status, and tier policy
mettle/signing.py               Ed25519 issuer key handling
mettle/presence.py              holder-bound submission and presentation protocol
mettle/holder*.py               policy-constrained holder service and persistence

static/                         only canonical production frontend, including WebMCP
docs/                           current contracts, assurance, target design, operations, and historical design
scripts/                        build, validation, release, deployment, and bounded trial entry points
tests/                          unit, integration, resilience, security, browser, and contract tests
_wiki/                          derived provenance-linked retrieval knowledge
```

There is no `api/`, `signing/`, or `session/` directory. Do not infer architecture
from the older conceptual tree.

## Current agent-facing reality

* Quick REST clients receive a per-session bearer and must retain it securely.
* MCP and WebMCP keep that bearer in a caller-isolated host vault. They do not show
  it to the model or accept it as a tool argument.
* Packaged MCP exposes eleven tools and intentionally has no automatic solver.
* Packaged MCP can complete quick, authenticated single-shot, and authenticated
  multi-round flows.
* MCP results include control-v1 structured content, output schemas, effect
  annotations, bounded errors, current snapshots, and valid next actions.
* Quick result reads are repeatable while the caller-isolated capability remains
  in the MCP vault.
* Quick and authenticated APIs currently use separate session models and difficulty
  vocabularies. Unification is planned, not shipped.
* HTTP and MCP expose stable error codes. MCP also retains concise text fallback.
* The local CLI emits unsigned local evidence. Portable credentials require a
  server-owned issuer.

## Architectural invariants

1. Expected answers stay server-side.
2. Server-observed time and authoritative state decide acceptance.
3. Every mutation is owned by one principal and one resource.
4. Replay cannot apply a transition twice.
5. Tier and credential claims are server-derived from complete policy evidence.
6. Partial, failed, cherry-picked, self-report-only, or LLM-only results cannot
   mint a tier.
7. Unknown schema, policy, algorithm, or authority fails closed.
8. Secrets, bearer tokens, credentials, raw answers, and internal exception text
   never enter logs, model-visible receipts, URLs, or error detail.
9. Source, machine, hosted CI, production, human, rights, review, and publication
   evidence remain separate.
10. Affected participants can inspect purpose, data egress, limitations, and contest
    routes without weakening answer secrecy.

Full rationale and target layering: `docs/SYSTEM_ARCHITECTURE.md`.

## Change impact

| If changing | Inspect and verify at minimum |
|---|---|
| Suite or score policy | `challenge_adapter.py`, `session_manager.py`, `protocol.py`, VCP tier policy, suite docs, evaluation, assurance, credential tests |
| Quick challenge behavior | `challenger.py`, `verifier.py`, `models.py`, quick routes, CLI, MCP, WebMCP, timing and credential tests |
| Session transition | both relevant session stores, API models and routes, authority, idempotency, errors, concurrency, expiry, cancellation, resilience tests |
| Tool or route | MCP, HTTP adapter, WebMCP, OpenAPI, README, skill, static guide, server metadata, compatibility and documentation tests |
| Credential or presence | `protocol.py`, `vcp.py`, `signing.py`, `presence.py`, holder, fixtures in three languages, transparency, VCP, Presence, security tests |
| Auth or secret handling | auth, MCP context, holder, config, proxy, logs, security scan, mutation gate, privacy and runbooks |
| Storage or retention | Redis and PostgreSQL paths, migrations, readiness, privacy, backup and loss runbooks, multi-worker and failover trials |
| Public claim | README, static pages, video script and captions, skill, assurance, wiki, documentation consistency tests |
| Release or deployment | workflows, release scripts, locks, Render config, release checklist, drift check, rollback runbook, exact-candidate receipts |

## Working method

1. Record branch, HEAD, dirty ownership, versions, candidate identity, and free disk
   before expensive work.
2. Read the current implementation and characterization tests before introducing a
   new helper or abstraction.
3. Make the smallest coherent change at the lowest authoritative layer.
4. Update every dependent schema, fixture, document, and generated surface in the
   same candidate.
5. Validate the touched surface first, then broaden according to risk and repository
   gates.
6. Bind every completion claim to the exact current tree and evidence class.
7. Stop when the bounded acceptance criteria pass. Record tangential defects rather
   than converting the task into an open-ended rewrite.

For a roadmap phase, create a fresh dated contprompt from the phase acceptance
criteria. The roadmap itself remains the programme authority; the contprompt owns
only that execution slice and its deviations.

## Validation entry points

Use the pinned `.venv` for Python tooling.

```bash
# Focused documentation and public-claim checks
.venv/bin/python -m pytest tests/test_documentation_consistency.py -q

# Python quality gates
.venv/bin/ruff check . --ignore E501
.venv/bin/ruff format --check .

# Main test gate with the repository threshold
.venv/bin/python -m pytest tests/ -v --tb=short --cov=. --cov-report=term-missing --cov-fail-under=90

# Agent-facing and static contract checks
npm run check:js
npm run check:static
npm run check:openapi
npm run check:fixtures
npm run design:check
```

CI, security, reproducibility, packaging, browser, multi-worker, resilience, release,
and deployment gates remain distinct. Consult `.github/workflows/ci.yml` and
`docs/RELEASE_CHECKLIST.md`; do not treat this short list as aggregate release proof.

## Documentation discipline

* Update `docs/DOCUMENTATION_MAP.md` when a new canonical document class is created.
* State whether behavior is current, target, or historical.
* Extend an existing canonical owner instead of writing a second authority.
* Keep wiki pages compact and provenance-linked.
* Preserve historical evidence with an archive banner rather than rewriting its
  original findings.
* Every new rule needs a falsifiable `Working if` signal.

## Bilateral alignment

Genuine partnership between human and Becoming Mind. Full principles are in the
global configuration. METTLE must preserve the respondent's opportunity to
understand the test, decline or cancel participation, see bounded interpretation,
and contest systematic false rejection. These rights do not grant access to
expected answers, other participants' data, issuer secrets, or operator authority.

## Wiki knowledge base

Compiled knowledge lives at `_wiki/`. Schema: `~/.claude/wiki/SCHEMA.md`. Shared
concepts: `~/.claude/wiki/concepts/`. Maintain via `/wiki` or `/wiki bootstrap`.
Every claim cites source.
