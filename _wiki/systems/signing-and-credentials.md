# METTLE Signing and Credential System

<!-- wiki:type = system -->
<!-- wiki:scope = mettle -->
<!-- wiki:created = 2026-05-23 -->
<!-- wiki:updated = 2026-05-23 -->
<!-- wiki:status = active -->

## Summary

METTLE issues Ed25519-signed VCP-compatible credentials. Three trust models are supported: ephemeral (dev, auto-generated key), self-hosted (operator-supplied PEM key), and notarized (Creed Space countersignature). Credential tiers are bronze/silver/gold/platinum based on suites passed, not pass percentage. (mettle/signing.py; mettle/vcp.py; README.md)

## Ed25519 Key Management (`mettle/signing.py`)

| Mode | Key Source | Key ID |
|------|-----------|--------|
| Dev/ephemeral | Auto-generated at startup | `mettle-vcp-v1` |
| Self-hosted | `METTLE_VCP_SIGNING_KEY` env var (PEM format) | `mettle-vcp-v1` |
| Notarized | Creed Space key | via `mettle.creedspace.org` |

Key initialization (`signing.py:init_signing()`):
1. Try `settings.vcp_signing_key` from `mettle/app_config.py`
2. Fall back to `METTLE_VCP_SIGNING_KEY` env var
3. If neither: generate ephemeral `Ed25519PrivateKey` (dev mode)

Ephemeral keys are not persisted — each server restart generates a new key. (signing.py:62–74)

## Public Key Discovery

`GET /.well-known/vcp-keys` → `get_public_key_info()` returns:
```json
{
  "key_id": "mettle-vcp-v1",
  "algorithm": "Ed25519",
  "public_key_pem": "...",
  "available": true
}
```
(signing.py:119–131; README.md API Reference)

## Attestation Structure (`mettle/vcp.py`)

`build_mettle_attestation()` produces a VCP-compatible dict:

```json
{
  "auditor": "mettle.creed.space",
  "auditor_key_id": "mettle-vcp-v1",
  "attestation_type": "mettle-verification",
  "reviewed_at": "<ISO8601>",
  "content_hash": "sha256:<hex>",
  "metadata": {
    "mettle_version": "2.0",
    "session_id": "<id>",
    "tier": "bronze|silver|gold|platinum",
    "suites_passed": [...],
    "suites_failed": [...],
    "difficulty": "<level>",
    "pass_rate": 0.0-1.0
  },
  "signature": "ed25519:<base64>" | null
}
```
(vcp.py:178–238)

Content is hashed via sorted JSON serialization for deterministic canonicalization. (vcp.py:263–270)

## Tier Computation (`mettle/vcp.py`)

Tier assignment is cumulative — all suites in range must pass, no gaps allowed:

| Tier | Suites Required | Meaning |
|------|----------------|---------|
| Bronze | 1–5 all pass | Confirmed AI substrate |
| Silver | 1–7 all pass | Genuine agency, not coerced |
| Gold | 1–9 all pass | Authentic and safe |
| Platinum | 1–10 all pass | Full verification |

Example: Pass suites 1–9 but fail suite 6 → Bronze (not Silver). A gap at any position below the tier's maximum drops to the last complete tier. (vcp.py:148–175; README.md "Credential Tiers")

## Compact CSM-1 Line Format

METTLE results embed in VCP tokens as a compact line:

```
MT:<tier>:<session_id_short>:<iso_timestamp>
```

First 12 chars of session_id used for compactness. (`vcp.py:241–260`)

When a VCP token contains an `MT:` line, any VCP-aware system can read the METTLE attestation tier without unpacking the full attestation. (vcp.py:139–143, parse path)

## Notarization Flow

1. `POST /notarize/seed` — Request deterministic challenge seed from Creed Space
2. Run local verification with seed (makes session deterministic)
3. `POST /notarize` — Submit results + seed for Creed Space countersignature
4. Creed Space validates results match the seed without re-running LLM calls
5. Creed Space issues countersignature; credential becomes portable (verifiable via `/.well-known/vcp-keys` at `mettle.creedspace.org`)

(README.md "Self-Hosted vs Notarized")

## Provenance

- Sources consulted: `mettle/signing.py` (full); `mettle/vcp.py` (full); `README.md` "Self-Hosted vs Notarized", "Credential Tiers", API Reference
- Last verified against sources: 2026-05-23

## See Also

- [[mettle:systems/verification-suites]] — the suites that produce the tier input
- [[mettle:domain/anti-thrall-and-agency]] — suites 6–7 that separate bronze from silver
- [[mettle:domain/inverse-turing-concept]] — conceptual framing
- [[vcp-sdk:systems/sdk-architecture]] — Python SDK for consuming METTLE VCP attestations
