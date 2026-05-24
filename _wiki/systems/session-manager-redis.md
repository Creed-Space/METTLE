# Session Manager and Redis State Machine

<!-- wiki:type = system -->
<!-- wiki:scope = mettle -->
<!-- wiki:created = 2026-05-23 -->
<!-- wiki:updated = 2026-05-23 -->
<!-- wiki:status = active -->

## Summary

`mettle/session_manager.py` manages METTLE verification session lifecycle backed by Redis. Sessions follow a state machine from creation through challenge generation to completion or expiry. Implements rate limiting, TTL management, and secure challenge/answer separation.

## State Machine

States: `CREATED → CHALLENGES_GENERATED → IN_PROGRESS → COMPLETED/EXPIRED/CANCELLED` (`session_manager.py:5`).

## Redis Keys and TTLs (`session_manager.py:26–34`)

| Key Pattern | TTL | Purpose |
|------------|-----|---------|
| `mettle:session:{id}` | 300s | Active session data |
| `mettle:session:{id}:*` | 3600s | Completed session data |
| `mettle:rate:{user_id}:{kind}` | 3600s | Rate limit window |

## Rate Limits (`session_manager.py:36–38`)

- `MAX_ACTIVE_SESSIONS_PER_USER = 5`
- `MAX_SESSIONS_PER_HOUR = 100`

## Session Creation (`session_manager.py:57–80`)

`create_session(user_id, suites, difficulty, entity_id, vcp_token)` returns `(session_id, client_challenges, session_metadata)`. Steps:

1. Check rate limits against both active-session and hourly counters
2. Resolve suite list (validates against `SUITE_REGISTRY` in `challenge_adapter.py`)
3. Generate `secrets.token_urlsafe(32)` session ID
4. Compute `expires_at = now + ACTIVE_SESSION_TTL`

The `vcp_token` parameter enables VCP-linked sessions where the entity's value context is carried into the verification flow.

## Challenge/Answer Separation

`ChallengeAdapter` (imported from `challenge_adapter.py`) returns `(client_data, server_answers)` pairs. Client challenges are stored in Redis and sent to the agent; server answers stay server-side and are never exposed unless the challenge is passed (anti-harvesting). See `systems/verifier-functions.md` for per-type verification.

## Suite Registry

Ten suites registered in `challenge_adapter.py:17–27`: adversarial, native, self-reference, social, inverse-turing, anti-thrall, agency, counter-coaching, intent-provenance, novel-reasoning.

## Provenance

- Sources: `mettle/session_manager.py:1–80`, `mettle/challenge_adapter.py:1–27`
- Last verified: 2026-05-23

## See Also

- [[mettle:systems/mcp-server-and-api]] — FastAPI endpoints that call SessionManager
- [[mettle:systems/verification-suites]] — ten procedurally generated suites
- [[mettle:systems/signing-and-credentials]] — notarization of completed sessions
