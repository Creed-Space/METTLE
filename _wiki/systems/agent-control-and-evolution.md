# Agent Control and System Evolution

<!-- wiki:type = system -->
<!-- wiki:scope = mettle -->
<!-- wiki:updated = 2026-08-31 -->
<!-- wiki:status = proposed -->

## Control Model

METTLE's proposed architecture applies one loop to runtime use and engineering
change: discover, plan, act under explicit authority, observe typed state, verify,
and retain a bounded receipt. The shared loop is intended to make both participant
sessions and candidate evolution legible to agents
(`docs/SYSTEM_ARCHITECTURE.md`; `docs/AGENT_CONTROL_PLANE.md`).

The target resource snapshot carries state, revision, time and quota budget,
available actions, limitations, and evidence. Each mutation uses an action or
revision precondition and returns the next snapshot. Secrets stay in a
caller-isolated host capability vault, while model-visible content receives a
non-secret handle (`docs/AGENT_CONTROL_PLANE.md`).

## Current Phase 1 Slice and Gaps

The 2026-08-31 source tree adds an eleven-tool MCP control-v1 subset with output
schemas, structured outcomes, bounded errors, snapshots, next actions, repeatable
quick-result reads, inspection, authenticated cancellation, and multi-round
control. Quick and authenticated state models and difficulty vocabularies remain
separate. Plans, revisions, idempotency, budgets, durable recovery, and
cross-transport generation remain future phases (`mettle/mcp_server.py`;
`mettle/mcp_contract.py`; `mettle/api_models.py`; `docs/ERROR_TAXONOMY.md`;
`docs/IDEMPOTENCY.md`).

## Evolution Plan

The active roadmap first closes incomplete and misleading paths, then adds a
common contract spine, safe retry and revisions, generated adapter parity, a
shared application kernel, and candidate-bound engineering control. Historical
audit plans remain evidence rather than execution authority
(`docs/AGENTIC_SYSTEM_ROADMAP.md`; `docs/DOCUMENTATION_MAP.md`).

## Evidence Boundary

Target design is never current runtime proof. Each phase requires exact-candidate
source and machine evidence plus the applicable hosted, production, human, rights,
independent-review, and publication gates
(`docs/ASSURANCE_CASE.md`; `docs/PROTOCOL_GOVERNANCE.md`).
