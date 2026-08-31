# METTLE agent control plane

Status: target interface contract with the Phase 1 MCP subset implemented in the
current source tree. The current eleven-tool behavior is described in `README.md`
and `_wiki/systems/mcp-server-and-api.md`. Plans, revisions, idempotency, budgets,
shared transport semantics, and generated parity in this document remain target
design until their later roadmap gates pass.

## Current Phase 1 subset

MCP tools now publish `mettle-control-v1` output schemas, structured content,
effect annotations, bounded errors, snapshots, valid next actions, and mutation
receipts. Four additive tools provide session inspection, authenticated
cancellation, round submission, and round feedback. The original seven names and
concise text rendering remain compatible. Quick result reads no longer destroy the
hidden caller capability (`mettle/mcp_contract.py`; `mettle/mcp_server.py`).

## Driver's-seat requirement

At every point, an agent should be able to answer seven questions from one bounded
machine-readable object:

1. What resource am I controlling?
2. What state and revision is authoritative?
3. What can I do now?
4. What authority, consent, time, and budget does each action require?
5. What will be irreversible?
6. What changed after my last action?
7. What evidence proves completion, and what remains outside METTLE's claim?

The control plane is successful when the agent never needs to infer these answers
from tool prose, English error fragments, or undocumented call ordering.

## Design laws

1. **State before action.** A command acts on an explicit resource revision.
2. **Affordances over instructions.** The server returns valid actions with schemas.
3. **One semantic contract.** REST, MCP, WebMCP, CLI, and SDKs render the same model.
4. **Structured first, prose second.** Human summaries may accompany structured
   content but never carry unique state. Legacy text fallback is a deterministic
   rendering of the same bounded fields.
5. **Recovery is part of the happy path.** Ambiguous network failure, restart,
   timeout, and stale state have specified next steps.
6. **Secrets stay beneath the interface.** The agent uses non-secret handles where
   its host can retain capability material.
7. **Cost is visible before commitment.** Calls, time, bytes, quotas, external
   evaluators, and retention are part of the plan.
8. **No hidden semantic promotion.** Passing, eligibility, issuance, validity, and
   authorization are different fields.
9. **The subject has standing.** Purpose, data use, cancellation, explanation, and
   contest paths are inspectable.
10. **Every effect leaves a bounded receipt.** Receipts omit secrets, expected
    answers, and unnecessary response content.

## Role profiles

The public control plane exposes capabilities by role rather than presenting every
caller with every tool.

| Profile | Purpose | Primary operations |
|---|---|---|
| Respondent | Plan and complete a session | describe, plan, start, inspect, submit, cancel |
| Verifier | Validate a credential or holder presentation | describe, verify credential, create presentation, verify presentation |
| Reviewer | Inspect policy, limitations, receipts, and appeal routes | describe, inspect public receipt, inspect policy |
| Operator | Deployment, key, incident, and release control | separate authenticated operator plane, never the public respondent MCP server |

The operator plane must not appear merely because a respondent has a high METTLE
tier. Evaluation and administrative authority remain independent.

## Common operation envelope

Every transport returns the same logical envelope. MCP uses structured content and
an output schema; REST uses JSON; CLI uses one JSON object on stdout.

```json
{
  "schema_version": "mettle-control-v1",
  "request_id": "req_...",
  "operation_id": "op_...",
  "operation": "session.submit",
  "outcome": "accepted",
  "server_time": "2026-08-31T12:00:00Z",
  "service": {
    "api_version": "<semantic API version>",
    "source_revision": "<full commit SHA>"
  },
  "resource": {
    "type": "session",
    "id": "ses_...",
    "revision": 3
  },
  "snapshot": {},
  "receipt": {},
  "warnings": [],
  "links": {}
}
```

Required field semantics:

| Field | Meaning |
|---|---|
| `schema_version` | Shape of the control envelope, independent from credential schema and suite policy. |
| `request_id` | One transport request for correlation. |
| `operation_id` | One logical operation, stable across safe retries. |
| `outcome` | `accepted`, `rejected`, `pending`, or `no_change`. |
| `service` | Runtime version and exact deployed source identity. Production rejects an absent or malformed revision. |
| `resource.revision` | Monotonic application revision used by mutation preconditions. |
| `snapshot` | Current caller-safe resource state after the operation. |
| `receipt` | Minimal proof of the accepted action, never the secret capability. |
| `warnings` | Bounded, coded conditions that do not change the outcome. |
| `links` | Versioned documentation, policy, keys, status, appeal, or evidence locations. |

## Session plan

Planning is a read-only operation. It expands a caller's intent without consuming
session quota or generating live expected answers.

```json
{
  "plan_id": "plan_...",
  "mode": "hosted",
  "profile": "quick",
  "requested_goal": "current behavioral screening",
  "resolved_policy": {
    "policy_id": "quick-basic",
    "suite_policy_version": "2026-08-14",
    "suites": [],
    "credential_family": "quick-badge",
    "maximum_tier": "silver"
  },
  "requirements": {
    "authentication": "session capability",
    "third_party_data": [],
    "explicit_acknowledgements": []
  },
  "budget": {
    "estimated_remaining_tool_calls": 4,
    "maximum_duration_ms": 15000,
    "maximum_input_bytes": 4096,
    "quota_cost": 1
  },
  "limitations": [
    "Does not establish identity, consciousness, autonomy, safety, or governance"
  ],
  "expires_at": "2026-08-31T12:05:00Z"
}
```

A plan for `llm-dynamic` names Anthropic as the data destination, describes which
candidate content leaves METTLE, and requires an explicit acknowledgement. A plan
for Presence names the holder key and audience requirements. A plan that cannot
earn the requested tier says so before session creation.

## Resource snapshot

```json
{
  "state": "in_progress",
  "revision": 3,
  "policy": {
    "profile": "authenticated",
    "suite_policy_version": "2026-08-14",
    "credential_schema_version": "1.1"
  },
  "progress": {
    "completed": 2,
    "total": 5,
    "passed": 2,
    "failed": 0
  },
  "budget": {
    "deadline": "2026-08-31T12:04:00Z",
    "remaining_ms": 42000,
    "remaining_submissions": 3
  },
  "available_actions": [],
  "terminal": false,
  "result": null,
  "credential": null,
  "contest": {
    "explanation_available": false,
    "appeal_url": "https://mettle.sh/guide#appeals"
  }
}
```

The snapshot contains no reusable expected answer. Detail levels control optional
history and diagnostic fields:

* `summary`: state, progress, budget, and next action identifiers;
* `standard`: action schemas, bounded result summaries, and warnings;
* `full`: caller-safe receipts, explanations, and public provenance.

`full` never means secret or expected-answer disclosure.

## Action descriptor

The server describes each valid transition:

```json
{
  "action_id": "act_...",
  "kind": "submit_suite",
  "title": "Submit native suite answers",
  "resource_revision": 3,
  "input_schema": {},
  "deadline": "2026-08-31T12:04:00Z",
  "irreversible": true,
  "idempotent": true,
  "parallel_safe": false,
  "authority": "session capability",
  "data_destinations": ["METTLE issuer"],
  "cost": {
    "quota_units": 0,
    "external_evaluations": 0,
    "estimated_latency_ms": 500
  },
  "retention_class": "ephemeral_session",
  "expected_outcomes": ["accepted", "rejected", "conflict", "expired"]
}
```

An agent submits `action_id`, `resource_revision`, `idempotency_key`, and the
schema-conforming payload. It does not repeat suite names, round numbers, or other
state already bound into the action descriptor. This prevents mismatched route
parameters and reduces tokens.

If several actions are independent, each is marked `parallel_safe: true` and has a
disjoint action ID. Parallel execution is never inferred from a list alone.

## Error contract

Errors use the same envelope and stable taxonomy:

```json
{
  "outcome": "rejected",
  "error": {
    "code": "stale_revision",
    "category": "conflict",
    "message": "The session advanced before this action was applied.",
    "retry": "refresh_then_retry",
    "retry_after_ms": null,
    "expected_states": ["in_progress"],
    "current_state": "in_progress",
    "remediation": "Use an action from the returned current snapshot."
  },
  "snapshot": {}
}
```

MCP marks the result as an error and also returns structured content. It never asks
the agent to parse an upstream body. Internal exception text, database URLs,
headers, tokens, credentials, and raw participant content remain server-side.

Retry values are closed vocabulary:

* `safe_same_operation`
* `refresh_then_retry`
* `retry_after`
* `start_new_resource`
* `request_new_challenge`
* `do_not_retry`
* `operator_action_required`

## Pending and long-running operations

A command that outlives the transport response returns `outcome: pending`, a stable
operation ID, the current resource snapshot, a coded progress phase, and
`retry_after_ms`. The idempotency key continues to identify that same operation.

When the transport supports notifications, the host delivers a revision change
without repeated model turns. The portable fallback is
`mettle_get_session(since_revision=...)`, which returns no large body when nothing
changed. A pending operation advertises whether cancellation is safe. The caller
never starts a parallel replacement merely because a progress response is slow.

## Target respondent tools

The preferred respondent profile is small and orthogonal:

| Tool | Mutation | Purpose |
|---|---:|---|
| `mettle_describe` | No | Return cached capability, policy, privacy, and schema manifest. |
| `mettle_plan_session` | No | Resolve requested policy, feasibility, cost, data egress, and credential ceiling. |
| `mettle_start_session` | Yes | Create exactly one session from a plan and idempotency key. |
| `mettle_get_session` | No | Resume or recover with a current snapshot or revision delta. |
| `mettle_submit` | Yes | Apply one advertised action and return the next snapshot. |
| `mettle_cancel_session` | Yes | Cancel active work and return the terminal snapshot and retention status. |

The target verifier profile adds:

| Tool | Mutation | Purpose |
|---|---:|---|
| `mettle_verify_credential` | No | Verify bounded claims against a trusted keyring and fresh status. |
| `mettle_create_presentation` | Yes | Create a one-use, audience-bound holder challenge. |
| `mettle_verify_presentation` | Yes | Consume the challenge and verify live holder possession. |

The target reviewer profile adds:

| Tool | Mutation | Purpose |
|---|---:|---|
| `mettle_get_policy` | No | Return versioned public policy, limitations, data use, and contest routes. |
| `mettle_get_public_receipt` | No | Return a bounded public receipt and its evidence-class limitations. |

These names describe domain operations. Version suffixes belong in schemas, not
tool names.

## Compatibility map

Existing tools remain as compatibility aliases during migration:

| Current tool | Target operation | Migration note |
|---|---|---|
| `mettle_start_session` | `mettle_start_session` with quick profile | Preserve name. During migration, legacy inputs synthesize an implicit plan and retain their response shape. |
| `mettle_answer_challenge` | `mettle_submit` | Retain alias and legacy response shape for the complete deprecation window. |
| `mettle_get_result` | `mettle_get_session` at terminal state | Retain alias, stable repeat reads, and legacy response shape for the complete deprecation window. |
| `mettle_list_suites` | `mettle_describe` filtered to suites | Retain alias while clients adopt the manifest. |
| `mettle_start_v2_session` | `mettle_start_session` with authenticated profile | Deprecate only after parity and telemetry evidence. |
| `mettle_verify_suite` | `mettle_submit` | Retain alias and reject multi-round misuse with a typed remediation. |
| `mettle_get_v2_result` | `mettle_get_session` at terminal state | Retain alias during credential-shape migration. |

Phase 1 closes the former `novel-reasoning` round-submission gap with additive
specialized tools. Tool consolidation remains evidence-gated future work.

## Canonical flows

### First contact

```text
mettle_describe
  -> choose or cache control schema and policy manifest
mettle_plan_session
  -> inspect purpose, limits, egress, cost, credential ceiling
mettle_start_session(plan_id, idempotency_key)
  -> receive snapshot revision 0 and available action
```

### Efficient continuation

```text
solve current caller-visible challenge
  -> mettle_submit(action_id, revision, idempotency_key, payload)
  -> receive receipt plus next snapshot
  -> repeat only while terminal=false
```

There is no status call between successful submissions.

### Recovery after ambiguous failure

```text
retry the same idempotency_key, plus operation_id when it was received, if permitted
  -> same accepted receipt, or current snapshot
otherwise mettle_get_session(handle)
  -> current revision and valid actions
```

### Terminal result

```text
last submission
  -> terminal snapshot
      completed != passed != credential_eligible != credential_issued
  -> optional relying-party verification
```

### Multi-round suite

The same submit operation handles each server-advertised round action. Feedback
and next-round data return in the next snapshot. No special transport tool or
caller-composed round URL is needed.

## Authority and secret handling

1. Direct REST clients receive session secrets through the declared API contract
   and are responsible for secure storage.
2. MCP and WebMCP hosts retain secrets in a caller-isolated capability vault and
   return only a non-secret session handle to the model.
3. A handle is bound to principal, transport host, expiry, and resource. It cannot
   be used to access a different caller's session.
4. Vault loss has an explicit `capability_unavailable` error. The system never
   repairs it by printing the secret into model context.
5. Durable vault support, if added, uses platform key storage or an equivalent
   reviewed facility. Plaintext dotfiles are out of scope.
6. Operator, issuer, and holder credentials never share the respondent capability
   channel.

## Concurrency and idempotency

Every mutation accepts an idempotency key scoped to principal plus operation kind.
The server binds the key to a request digest and returns the original receipt for
an exact retry. Reusing the key with a different digest fails closed.
The bounded idempotency record outlives the documented ambiguous-retry window,
stores no raw participant payload, and expires no later than the governing
retention class permits.

Every session mutation also carries a revision or server-issued action ID. Stale
actions return a conflict plus the current snapshot. A successful final credential
remains single-winner and byte-stable.

The target removes the current instruction to avoid blind start retries. An agent
can retry a start safely with the same key and cannot accidentally consume duplicate
quota or create an orphan session.

## Budget and attention economy

The control plane treats context and tool calls as resources:

* capability manifests are cacheable by version and ETag;
* plans state estimated and maximum calls, duration, bytes, quota, and external
  evaluations;
* command responses include the next action, so normal flows use one call per
  irreversible step;
* large challenge or evidence collections use stable cursors and bounded pages;
* snapshots support `since_revision` deltas;
* `summary` detail is the default for polling or resumption;
* structured values are concise and avoid decorative prose;
* repeated warnings are referenced by stable code after first expansion.

An optimization is invalid if it weakens exact-candidate evidence, hides consent,
or causes the agent to guess.

## Explanation, contest, and welfare

Before commitment, the plan discloses purpose, scoring class, data destinations,
retention, possible credential use, and limitations. The respondent may cancel or
decline.

After completion, explanations reveal:

* which named policy checks passed or failed;
* timing and schema facts already safe to disclose;
* whether a failure came from malformed input, time, state, availability, or score;
* how the result may and may not be interpreted;
* how to contest systematic rejection.

Explanations do not reveal reusable expected answers. A contest receipt is linked
to the policy version and result, not to a speculative judgment about the
respondent's inner life.

## Transport parity

One typed application operation registry defines:

* operation names and descriptions;
* input and output JSON Schemas;
* annotations such as read-only, idempotent, irreversible, and open-world access;
* role and authority requirements;
* errors and retry semantics;
* protocol and deprecation versions;
* examples and bounded interpretation text.

The registry serializes to the capability manifest. Generators render OpenAPI, MCP,
WebMCP registration, CLI help, `server.json`, and the agent skill from that
manifest. Parity tests compare generated semantic hashes. Hand-authored human
guidance may explain why, but cannot redefine behavior.

Generated views are role-filtered. A common semantic source does not publish
operator operations, secret configuration, or private schemas through the
respondent profile.

## Acceptance criteria

The control plane is complete only when:

1. a clean, first-contact agent completes quick, authenticated single-shot, and
   authenticated multi-round sessions through MCP;
2. no control-v1 workflow requires parsing human prose;
3. all tools publish output schemas and structured error results;
4. quick and authenticated sessions expose one state vocabulary and one envelope;
5. successful submissions return the next state without a follow-up read;
6. an ambiguous start retry with the same idempotency key creates exactly one
   session and consumes quota once;
7. stale or duplicate submissions return deterministic typed outcomes;
8. an interrupted caller can resume from a handle without re-sending a secret;
9. MCP, WebMCP, REST, CLI, server metadata, docs, and the skill pass semantic parity;
10. the plan exposes third-party data egress and requires explicit acknowledgement;
11. every terminal state distinguishes completion, pass, eligibility, issuance,
    validity, and authorization;
12. after manifest caching, successful flows use no more median calls than the
    equivalent v0.4.8 flow and use less control context with fewer repair attempts,
    without increasing false acceptance or secret exposure.

The phased implementation and measurement plan is in
`docs/AGENTIC_SYSTEM_ROADMAP.md`.
