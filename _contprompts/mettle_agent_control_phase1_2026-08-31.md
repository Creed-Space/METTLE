---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8, 9]
verification_criteria:
  - "Quick, authenticated single-shot, and authenticated multi-round MCP journeys are machine-readable and complete."
  - "Repeated quick-result reads are stable and never expose the hidden bearer."
  - "MCP failures set isError and return bounded structured control errors."
  - "The seven v0.4.8 tool names remain available and no automatic solver is exposed."
  - "Focused security tests and complete applicable repository gates pass."
---

# METTLE Agent Control Phase 1

## Authority

Nell released the implementation hold with "Proceed please" on 2026-08-31.
This contprompt governs Phase 1 of `docs/AGENTIC_SYSTEM_ROADMAP.md`. It does not
authorize a commit, push, release, deployment, registry publication, or production
verification.

## Exact-current baseline

* Repository: `/Users/nellwatson/Documents/GitHub/METTLE`
* Source HEAD: `ed7b429a40e2049146dbf3a165ace1d4e529e463`
* Source tag: `v0.4.8`
* Branch: `main`, aligned with `origin/main` at preflight
* Existing working tree: the uncommitted Phase 0 documentation and design change
  listed in the 2026-08-31 task status. Preserve it intact.
* Free disk at preflight: approximately 8.7 GiB

## Goal

Make each advertised MCP verification workflow complete and machine-readable with
the smallest coherent additive runtime change. Preserve the current application
kernel and compatibility text while giving an agent typed state, typed outcomes,
bounded errors, valid next actions, inspection, cancellation, and multi-round
control.

## Adversarial pre-check

### Scope risks examined

1. A common application-kernel refactor would exceed Phase 1 and raise regression
   risk. Phase 1 will adapt the existing REST authority through the MCP boundary.
2. Replacing the seven legacy tools would violate compatibility. New operations
   will be additive and existing names will retain concise text fallback.
3. A generic submit operation would pull Phase 2 action IDs and idempotency into
   this slice. Phase 1 will add explicit round operations and keep the current
   single-shot operation.
4. Quick cancellation has no current REST authority. The new cancellation tool
   will accurately declare and support authenticated sessions only. It will not
   pretend that a quick-session cancellation path exists.
5. Upstream response bodies and exception strings are currently unbounded. The
   adapter must classify failures by status and emit a fixed safe message without
   forwarding the body.
6. Consuming the quick-session bearer after one result read breaks recovery. The
   capability stays in the caller-isolated vault until its existing TTL expires.
7. Full repository validation can be storage intensive. Run focused checks first,
   then one applicable complete gate sequence if disk remains adequate. Never
   repeat an unchanged expensive gate without new evidence.

### Surviving ambiguities

None changes the Phase 1 architecture. Unified submit, operation revisions,
idempotency keys, plan objects, and generated cross-transport manifests remain in
later roadmap phases.

## Implementation plan

1. Add a versioned MCP control-contract module containing JSON Schemas, outcome
   builders, session snapshots, next-action derivation, bounded error mapping, and
   compatibility text support.
2. Annotate every MCP tool and publish an output schema.
3. Return structured content and protocol `isError` through the low-level MCP
   handler while retaining iterable text content for existing in-process clients.
4. Add authenticated session inspection and cancellation tools.
5. Add multi-round submission and completed-round feedback tools.
6. Make every mutation return a current snapshot, receipt, and valid next actions.
7. Keep quick result authority available for repeat reads.
8. Add first-contact, security-boundary, annotation, schema, error, recovery, and
   end-to-end adapter tests. Update installed-surface and documentation assertions.
9. Update current-behavior documentation and the roadmap evidence ledger.

## Definition of Done

* The MCP tool list contains the original seven tools plus the four bounded Phase 1
  controls: session inspection, authenticated cancellation, round submission, and
  round feedback.
* All tools publish output schemas and accurate read-only, destructive,
  idempotent, and open-world annotations.
* Successful calls include `schema_version`, `operation`, `outcome`, a typed data
  payload, a session snapshot where applicable, valid next actions, and a receipt
  for mutations.
* Failures return `isError: true` with a bounded structured error code, safe
  message, retry guidance, and HTTP status when one exists.
* No raw upstream response body, internal exception text, API key, session token,
  expected answer, or automatic solver crosses the MCP boundary.
* Quick results can be read repeatedly during the capability TTL.
* Quick, authenticated single-shot, and authenticated multi-round acceptance
  journeys pass using only listed tool schemas and returned structured content.
* Existing text consumers continue to receive concise useful fallback content.
* Focused tests, Ruff, formatting, mypy, packaging smoke, frontend checks where
  applicable, and the complete pytest gate pass on the exact working tree, or any
  external limitation is reported with its exact evidence class.
* Documentation states the exact implemented surface and keeps later target work
  clearly marked as proposed.

## Non-goals

* No Phase 2 idempotency, revisions, action IDs, polling deltas, or budgets.
* No Phase 3 generated operation registry or WebMCP and CLI convergence.
* No application-kernel rewrite.
* No quick-session cancellation claim without a supporting REST operation.
* No release, deployment, or publication.

## Deviations

| Time | Observation | Conservative choice | Effect |
|---|---|---|---|
| 2026-08-31 | The existing `.venv` contained MCP 1.29.0 while the hash-locked MCP requirements specify 2.0.0. | Synchronize the existing virtual environment from `requirements-mcp-lock.txt` before judging compatibility. | Exact repository gates ran on the pinned MCP 2.0.0 dependency set. The clean wheel smoke additionally exercised the declared package range with MCP 2.1.1. |
| 2026-08-31 | A mutation followed by a status read could commit successfully and then surface a retryable read failure. | Read authenticated state before verification or round submission, then derive the post-mutation snapshot from the successful response. Derive cancellation's terminal snapshot from the successful DELETE. | A caller is never told to retry an already committed mutation solely because a follow-up inspection failed. |
| 2026-08-31 | Existing GET routes can issue or persist stable result material. | Mark result and session inspection operations conservatively as non-read-only while retaining their idempotent, non-destructive annotations. | Host policy sees the real server-side effect instead of a misleading read-only claim. |
| 2026-08-31 | The security mutation gate anchored the former seven-tool `list_tools` source shape. | Update the exact anchor to the eleven-tool schema-bearing implementation, then rerun the full mutation gate. | All 9 security mutations are killed against the current source shape. |
| 2026-08-31 | A first-contact multi-round client must submit every challenge key issued for the round. | Derive the answer object's top-level keys solely from the structured start response. | The acceptance test proves schema-visible orchestration without hidden implementation knowledge or an automatic solver. |
| 2026-08-31 | Final review found that generic timeout and 5xx guidance could invite blind repetition of a mutation whose commit state was unknown. | Keep same-operation retry only for reads. Require refresh before retry for mutations with a session handle and operator action for ambiguous creation. | Phase 1 never represents an uncertain non-idempotent mutation as safely repeatable. |
| 2026-08-31 | Final review found that raw round values were interpolated into authenticated request paths after relying on host-side schema enforcement. | Validate exact integer type and the 1 to 5 range inside the adapter before path construction. Validate the authenticated session identifier returned at creation before publishing actions. | Direct and non-validating MCP hosts cannot use round arguments for path manipulation or publish an unsafe session handle. |

## Completion evidence

* Focused control-contract acceptance: 19 passed.
* Complete Python gate: 1,983 passed, 2 SQLite resource warnings,
  90.17% aggregate coverage.
* Security mutation gate: 9 of 9 mutations killed. Bandit reports no findings in
  the new MCP runtime modules.
* Ruff, format checking, mypy, Vulture, Cargo compatibility, static JavaScript,
  OpenAPI, fixture, frontend design, documentation, and Markdown checks pass.
* Dirty-candidate wheel and source distribution pass Twine validation. A clean
  wheel environment lists all 11 approved tools.
* Source and local machine acceptance are complete. Human review, hosted CI,
  release, deployment, registry publication, and production evidence remain
  intentionally unclaimed.
