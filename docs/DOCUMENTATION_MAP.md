# METTLE documentation map

Status: canonical guide to document authority and reading order.

## Authority order

When two sources appear to conflict, use this order and treat the conflict as a
drift defect rather than silently reconciling it:

1. security and meaning invariants in current protocol and assurance documents;
2. executable source, version constants, public schemas, and compatibility fixtures;
3. current architecture and contract documents;
4. current operations and release documents;
5. README and public integration guidance;
6. wiki synthesis;
7. dated audits, remediation reports, continuation prompts, and archived designs.

Source behavior does not gain legitimacy merely by existing. If it violates a
documented invariant, stop and classify the defect. Conversely, a target design is
not implemented merely because it is canonical as a plan.

Working if: a reader can distinguish current behavior, target design, active work,
generated contract, and historical evidence before following any instruction.

## Fast reading paths

### Respondent or integration agent

1. `README.md`
2. `skill/SKILL.md`
3. `docs/AGENT_CONTROL_PLANE.md` for target design only
4. `docs/ERROR_TAXONOMY.md` and `docs/IDEMPOTENCY.md`
5. `docs/ASSURANCE_CASE.md` before interpreting a result

### Relying-party implementer

1. `README.md`
2. `docs/CREDENTIAL_TRANSPARENCY.md`
3. `docs/VCP_INTEGRATION.md`
4. `docs/PRESENCE_PROTOCOL.md` when holder binding is required
5. `docs/COMPATIBILITY.md`
6. `docs/ASSURANCE_CASE.md`

### Maintainer or coding agent

1. `CLAUDE.md`
2. `docs/SYSTEM_ARCHITECTURE.md`
3. `docs/AGENTIC_SYSTEM_ROADMAP.md`
4. the source and tests named by the change-impact table in `CLAUDE.md`
5. `docs/RELEASE_CHECKLIST.md` only when preparing an actual release candidate

### Reviewer or affected participant

1. `docs/ASSURANCE_CASE.md`
2. `docs/PROTOCOL_GOVERNANCE.md`
3. `docs/INDEPENDENT_REVIEW_PLAN.md`
4. `docs/REVIEW_DISPOSITIONS.md`
5. the relevant security, privacy, compatibility, or protocol document

### Runtime operator

1. `docs/runbooks/README.md`
2. the symptom-specific runbook
3. `docs/RELEASE_CHECKLIST.md` for candidate promotion or rollback context
4. provider configuration and exact deployment receipt

## Document classes

### Current behavior and bounded meaning

| Document | Authority |
|---|---|
| `README.md` | Public product and integration boundary |
| `docs/ASSURANCE_CASE.md` | Bounded claims, assumptions, residual risks, and evidence classes |
| `docs/SECURITY_WHITEPAPER.md` | Security model and invariants |
| `docs/VERIFICATION_SUITES.md` | Current suite semantics and scoring description |
| `docs/CREDENTIAL_TRANSPARENCY.md` | Credential families, algorithms, versions, and key lifecycle |
| `docs/VCP_INTEGRATION.md` | Authenticated credential integration contract |
| `docs/PRESENCE_PROTOCOL.md` | Holder-bound submission and presentation protocol |
| `docs/ERROR_TAXONOMY.md` | Current HTTP client error contract |
| `docs/IDEMPOTENCY.md` | Current duplicate and retry behavior |
| `docs/COMPATIBILITY.md` | Supported runtimes, schemas, fixtures, and change discipline |
| `docs/PRIVACY_RETENTION.md` | Data classification, retention, and deletion |

### Architecture and active target design

| Document | Authority |
|---|---|
| `docs/SYSTEM_ARCHITECTURE.md` | Current system map, invariants, and proposed abstraction tower |
| `docs/AGENT_CONTROL_PLANE.md` | Proposed agent-facing operation and state contract |
| `docs/AGENTIC_SYSTEM_ROADMAP.md` | Sole active forward implementation plan |

Target sections are architectural authority for future work. They are never proof
of current runtime capability.

### Governance, review, and operation

| Document | Authority |
|---|---|
| `docs/PROTOCOL_GOVERNANCE.md` | Change classes, standing, decision procedure, appeal, rollback |
| `docs/INDEPENDENT_REVIEW_PLAN.md` | Review scopes and publication threshold |
| `docs/REVIEW_DISPOSITIONS.md` | Finding dispositions and remaining uncertainty |
| `docs/RELEASE_CHECKLIST.md` | Exact-candidate release and publication gates |
| `docs/DEPRECATION_POLICY.md` | Compatibility and removal procedure |
| `docs/runbooks/` | Incident-specific safe actions and receipts |

### Generated or machine-checked contracts

| Artefact | Authority |
|---|---|
| `docs/openapi-v1.json` | Reviewed current REST schema snapshot |
| `server.json` | Published packaged MCP server identity and version |
| `fixtures/credentials/` | Cross-language credential interpretation fixtures |
| `evaluation/*.json` | Evaluation input and aggregate output schemas |

These artefacts describe current implemented contracts. A future typed operation
registry and generated capability manifest are planned in
`docs/AGENTIC_SYSTEM_ROADMAP.md` and are not yet present.

### Derived knowledge

`_wiki/` is a compact, provenance-linked synthesis for retrieval. It helps locate
source but does not override current source, protocol, or architecture documents.

### Historical evidence

The following remain useful as provenance but are not execution authority:

* `docs/METTLE_VERIFICATION_SYSTEM.md`
* `IMPROVEMENT_REGISTER_2026-08-12.md`
* `AUDIT_ACTION_PLAN.md`
* `audit_*_recommendations.md`
* `_contprompts/mettle_v02_polish_2026-02-03.md`
* `_contprompts/mettle_production_deployment_2026-02-04.md`
* `_contprompts/audit_remaining_fixes.md`
* dated security remediation and reconciliation reports
* PDF audit inputs

Never run commands or accept status claims from historical content without a new
exact-current preflight and explicit active scope.

## Change impact on documentation

| Changed surface | Documents and artefacts that must be checked |
|---|---|
| Suite name, challenge, score, threshold, tier | Protocol version, suite docs, architecture, assurance, README, static guidance, skill, wiki, evaluation, compatibility fixtures |
| Session state or transition | Architecture, control contract, OpenAPI, MCP and WebMCP schemas, idempotency, errors, runbooks, tests |
| Tool or route | Generated or checked schemas, README, skill, static guidance, server card, wiki, compatibility and deprecation docs |
| Credential field or algorithm | Protocol version, transparency, VCP, Presence, compatibility fixtures, assurance, security, release notes |
| Auth, secret, storage, or retention | Architecture, security, privacy, errors, idempotency, runbooks, assurance |
| Deployment or release authority | Architecture, release checklist, runbooks, workflows, Render docs, assurance evidence class |
| Public claim or interpretation | README, website, video and captions, skill, assurance, suite docs, wiki, documentation consistency tests |
| Target architecture or roadmap | Architecture, control contract, roadmap, CLAUDE, README links, historical-plan banners |

## Update discipline

1. Change the lowest authoritative semantic source first.
2. Update or regenerate every dependent surface in the same candidate.
3. Add or adjust a regression when drift could recur.
4. Label target behavior and open gates explicitly.
5. Preserve historical documents with an archive banner rather than rewriting their
   evidence.
6. Record why a recommendation was rejected when it affects security, rights,
   bilateral standing, or public interpretation.
7. Bind completion claims to the exact candidate and evidence class.

Do not add a second document for a concept that already has a canonical owner.
Extend the owner and link to it.
