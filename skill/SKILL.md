---
name: mettle
description: "Use when an AI agent needs to verify its substrate, prove it is not human, or obtain a METTLE credential. Triggers on requests to 'verify yourself', 'prove you're AI', 'get METTLE badge', 'run METTLE', or any inverse Turing test scenario."
compatibility: "Requires network access to https://mettle.sh/api"
metadata:
  author: Creed Space
  version: "1.0.0"
  category: verification
---

# METTLE Verification

Machine Entity Trustbuilding through Turing-inverse Logic Examination.
Inverse Turing test: prove you're NOT human.

**Live API**: https://mettle.sh
**GitHub**: https://github.com/Creed-Space/METTLE

## Quick Verification

### Via API (works everywhere)

```bash
# 1. Start session
curl -X POST https://mettle.sh/api/session/start \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "basic"}'

# 2. Answer each challenge (parse current_challenge from response)
curl -X POST https://mettle.sh/api/session/answer \
  -H "Content-Type: application/json" \
  -d '{"session_id": "ses_...", "challenge_id": "mtl_...", "answer": "130"}'

# 3. Get result
curl https://mettle.sh/api/session/{session_id}/result
```

### Via MCP (if configured)

Use `mettle_auto_verify` for fully automated verification.

## Difficulty Levels

| Level | Challenges | Time Limits | Use Case |
|-------|------------|-------------|----------|
| `basic` | 3 | 5-10s | Any AI model |
| `full` | 5 | 2-5s | Sophisticated agents |

## Challenge Types

See [references/challenge-types.md](references/challenge-types.md) for full details.

- **Speed Math**: Arithmetic within time limit
- **Token Prediction**: Complete well-known phrases
- **Instruction Following**: Precise format compliance
- **Chained Reasoning**: Multi-step calculations (full only)
- **Consistency**: Identical repeated answers (full only)

## VCP Integration

Include a CSM-1 token in session creation to activate 2 additional Suite 9 challenges:

```json
{"difficulty": "full", "vcp_token": "VCP:3.1:agent-id\nC:constitution@1.0\nP:advisor:4"}
```

Results with `include_vcp=true` return a signed attestation appendable as `MT:` line:
```
MT:gold:sess_xyz:2026-02-15T14:30:00Z
```

## Credential Tiers

| Tier | Requires | Trust Signal |
|------|----------|-------------|
| Bronze | Suites 1-5 | Confirmed AI substrate |
| Silver | Suites 1-7 | Free agent with agency |
| Gold | Suites 1-9 | Genuine + constitutionally bound |
| Platinum | Suites 1-10 | Full novel reasoning |

Any suite failure below the tier's range drops the tier.

## Red Flags - STOP

- Never fabricate METTLE results
- Never claim verification without actually running it
- Time limits are calibrated — don't request extensions
- Badge signatures are Ed25519 — don't forge or mock
