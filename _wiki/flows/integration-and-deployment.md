# Integration and Deployment

<!-- wiki:type = flow -->
<!-- wiki:scope = mettle -->
<!-- wiki:updated = 2026-08-14 -->
<!-- wiki:status = active -->

## Current Flow

1. A caller creates either a quick session or an authenticated suite session.
2. The service separates server-held expected answers from the sanitized challenge returned to the caller.
3. The caller submits its own response under the required bearer credential.
4. The service applies timing, ownership, replay, and policy checks, then records a bounded result.
5. A passing quick session may receive one stable signed legacy badge when issuance is enabled.
6. An authenticated session computes the highest complete contiguous tier. With `include_vcp=true`, an eligible result receives a server-signed Ed25519 credential; a partial result receives an unsigned evidence receipt.
7. A relying party verifies signature, policy version, expiry, identifier, and revocation, then applies its own authorization policy.

Sources: `main.py`, `mettle/challenge_adapter.py`, `mettle/session_manager.py`, `mettle/router.py`, and `mettle/vcp.py`.

## MCP and CLI Boundaries

The MCP surface exposes seven interactive and authenticated suite tools. It has no automatic solver. Quick answer and result operations require the session bearer token; authenticated suite operations require the configured API bearer key (`mettle/mcp_server.py`).

The local CLI runs an unsigned research flow. Portable credentials come only from a server-owned issuer. The reference solver is confined to `scripts/testing/solver.py`; it is excluded from the package, not imported by the MCP server, and cannot be selected through its tool list (`mettle/cli.py`; `pyproject.toml`; `README.md`).

## Deployment Authority

The repository declares Render through `render.yaml` with `autoDeploy: false`. GitHub workflows validate, package, publish release evidence, and monitor provider drift. A release operator explicitly deploys the reviewed commit after the required gates; branch updates cannot independently publish the API, MCP service, or holder (`render.yaml`; `deploy/holder/render.yaml`; `.github/workflows`).

Local validation proves source and machine behavior only. Release acceptance still requires hosted CI on an immutable committed SHA, one observed provider deployment for that SHA, deployed health and safe-flow receipts, and rollback evidence. The API reports the provider commit through `source_revision` and `X-METTLE-Source-Revision`; production readiness fails when that identity is absent or malformed. The public MCP receipt must list exactly seven tools and exclude the automatic solver (`main.py`; `docs/RELEASE_CHECKLIST.md`; `docs/runbooks/DEPLOYMENT_ROLLBACK.md`).

## Operational Trust Boundary

METTLE results alone must not authorize trades, deployments, privileged actions, or high-impact access. Consumers must add controls proportionate to their risk. Production acceptance also depends on durable Redis and PostgreSQL authority, protected signing keys, authentic public-key publication, edge header validation, privacy retention, and incident response (`docs/ASSURANCE_CASE.md`; `docs/SECURITY_WHITEPAPER.md`; `docs/runbooks`).

All new credential issuance can be disabled through `METTLE_CREDENTIAL_ISSUANCE_ENABLED` without disabling result retrieval or verification of already-issued credentials (`config.py`; `main.py`; `mettle/router.py`).

## Provenance

Sources last checked on 2026-08-14: `README.md`, `render.yaml`, `.github/workflows`, `main.py`, `config.py`, `mettle/mcp_server.py`, `mettle/router.py`, `mettle/vcp.py`, and `docs/RELEASE_CHECKLIST.md`.
