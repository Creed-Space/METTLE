# Session Manager and Redis State Machine

<!-- wiki:type = system -->
<!-- wiki:scope = mettle -->
<!-- wiki:created = 2026-05-23 -->
<!-- wiki:updated = 2026-08-14 -->
<!-- wiki:status = active -->

## Summary

`mettle/session_manager.py` is the authoritative state manager for the authenticated twelve-suite API. It stores sessions, server-held answers, round state, results, presentation challenges, active reservations, and rate counters in Redis. Production readiness treats Redis loss as a security dependency failure (`mettle/session_manager.py`; `main.py`; `config.py`).

## Lifecycle and Expiry

The states are `CREATED`, `CHALLENGES_GENERATED`, `IN_PROGRESS`, `COMPLETED`, `EXPIRED`, and `CANCELLED`. Active sessions expire after 300 seconds and completed sessions after 3600 seconds. A completed result retains its original completion time, which is also the evidence time used for delayed credential issuance (`mettle/session_manager.py:45-55`; `mettle/session_manager.py:create_session`; `mettle/router.py:get_result`).

## Atomic Quotas and Mutation Locks

Session creation reserves the per-user active and hourly quotas in one Redis Lua operation after suite validation. Expired active reservations are removed by score before counting. Failed or cancelled creation releases its reservation. Security-relevant session mutation runs under an owner-token lock whose lease is refreshed until the operation completes; release deletes only the matching token (`mettle/session_manager.py:_RATE_RESERVATION_SCRIPT`; `mettle/session_manager.py:_reserve_rate_limits`; `mettle/session_manager.py:_session_lock`).

## Challenge and Answer Separation

Public challenges and server-held evaluation material use separate Redis keys. Single-shot results expose bounded verdict details. Novel-reasoning rounds expose only the current round and release the next round after an accepted transition. No public result includes reusable expected answers (`mettle/session_manager.py:_create_reserved_session`; `mettle/session_manager.py:verify_single_shot`; `mettle/session_manager.py:submit_round`; `tests/test_security_scan_20260814.py`).

## LLM and Presence Boundaries

Selecting `llm-dynamic` requires `allow_third_party_llm=true`; otherwise session creation fails before quota reservation. Presence mode adds holder-key proof and per-submission proof checks. Presence credentials additionally require authenticated online status acceptance. Neither mechanism establishes model identity or co-location of signer and solver (`mettle/session_manager.py:create_session`; `mettle/presence.py`; `mettle/vcp.py`).

## Provenance

Sources last checked on 2026-08-14: `mettle/session_manager.py`, `mettle/challenge_adapter.py`, `mettle/presence.py`, `mettle/router.py`, `main.py`, and `config.py`.
