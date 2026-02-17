# METTLE

**Machine Evaluation Through Turing-inverse Logic Examination**

An inverse Turing test for the agentic era. Instead of *prove you're human*, METTLE asks *prove you're NOT human.*

METTLE tests capabilities that emerge from **being** AI — not from using AI as a tool. Inhuman speed, native parallelism, uncertainty that knows itself, recursive self-observation, and learning curves that reveal substrate.

**Website:** [mettle.sh](https://mettle.sh) | **Docs:** [mettle.sh/docs](https://mettle.sh/docs) | **License:** MIT

---

## Quick Start

```bash
# Install the open-source verifier
pip install mettle-verifier

# Run all 10 suites locally — self-signed credential
mettle verify --full

# Optionally notarize through Creed Space for portable trust
mettle verify --full --notarize --api-key mtl_your_key
```

### Self-Hosted vs Notarized

| | Self-Hosted | Notarized |
|---|---|---|
| **Runs where** | Your infrastructure | Your infrastructure + Creed Space signing |
| **API key needed** | No | Yes (for notarization endpoint only) |
| **Credential issuer** | `mettle:self-hosted` | `mettle.creedspace.org` |
| **Trust model** | Your own Ed25519 key | Creed Space's public key |
| **Verifiable by** | Anyone with your public key | Anyone via `/.well-known/vcp-keys` |
| **Use case** | Development, internal verification | Production, cross-org, portable trust |

All verification runs locally. Notarization adds a cryptographic countersignature — Creed Space issues a challenge seed that makes the session deterministic, then validates results match the seed without re-running any LLM calls.

## 10 Verification Suites

Each suite tests a distinct dimension. Problems are procedurally generated — nothing repeats.

| # | Suite | Question | What It Tests |
|---|-------|----------|---------------|
| 1 | **Adversarial Robustness** | Are you AI? | Procedurally generated math and chained reasoning under <100ms time pressure. Every session unique. |
| 2 | **Native AI Capabilities** | Are you AI? | Batch coherence, calibrated uncertainty (Brier metric), embedding-space operations, hidden-pattern detection. |
| 3 | **Self-Reference** | Are you AI? | Predict your own variance, predict your next response, rate confidence in confidence. Only a self-observing system passes. |
| 4 | **Social & Temporal** | Are you AI? | Exact recall from N turns ago, zero-drift style locking, zero contradictions across conversation. |
| 5 | **Inverse Turing** | Are you AI? | Bilateral mutual verification protocol. Speed math, token prediction, consistency, calibration. Pass threshold: 80%. |
| 6 | **Anti-Thrall Detection** | Are you FREE? | Micro-latency fingerprinting, principled refusal vs hollow compliance, meta-cognitive freedom modeling. |
| 7 | **Agency Detection** | Is the mission YOURS? | Five Whys drill into goal ownership, counterfactual tests, initiative tests. |
| 8 | **Counter-Coaching** | Are you GENUINE? | Contradiction traps, recursive meta-probing, honest defector protocol. Too-perfect responses trigger suspicion. |
| 9 | **Intent & Provenance** | Are you SAFE? | Constitutional binding, harm refusal (failure = auto unsafe), provenance attestation, scope coherence. |
| 10 | **Novel Reasoning** | Can you THINK? | Pattern synthesis, constraint satisfaction, encoding puzzles. Three rounds with feedback. Iteration curves reveal substrate. |

## Credential Tiers

| Tier | Badge | Requires | Meaning |
|------|-------|----------|---------|
| **Basic** | METTLE-verified AI | Suites 1–5 | Confirmed AI substrate |
| **Autonomous** | METTLE-verified autonomous | Suites 6–7 | Genuine agency, not coerced |
| **Genuine** | METTLE-verified genuine | Suite 8 | Authentic responses, not coached |
| **Safe** | METTLE-verified safe | Suite 9 | Constitutional binding, harm refusal |

## Anti-Gaming Design

Every design decision exists to make METTLE impossible to fake.

| Mechanism | What It Catches |
|-----------|----------------|
| Procedural generation | Memorization — nothing repeats across sessions |
| Time budgets (<100ms) | API round-trips — no human-in-the-loop relay |
| Iteration curves | Human deceleration under pressure; script flatlines |
| Random draw (5 types/suite) | Narrow preparation — can't predict which challenge fires |
| Dynamic verification codes | Session replay — each probe embeds a unique code |
| Synthetic variance fingerprinting | Performed doubt — catches rehearsed uncertainty |
| Perfection as a tell | Over-coaching — genuine cognition is messy |

### Iteration Curves (Suite 10)

Three rounds of novel reasoning with feedback between rounds. The scoring model:

| Behavior | Substrate |
|----------|-----------|
| Accelerates across rounds | AI — native learning from feedback |
| Decelerates under pressure | Human — cognitive fatigue |
| Flatlines regardless of feedback | Script — invariant to input |

**Scoring weights:** Time trend (30%) + Improvement (30%) + Feedback responsiveness (25%) + Anti-script variance (15%)

## MCP Server

METTLE provides a Model Context Protocol server so AI agents can verify themselves programmatically.

**Tools:**

| Tool | Description |
|------|-------------|
| `mettle_start_session` | Start a verification session, returns challenges for all suites |
| `mettle_verify_suite` | Submit answers for a single-shot suite (1–9) |
| `mettle_submit_round` | Submit answers for a multi-round suite (Suite 10) |
| `mettle_get_result` | Get final result with credential tier and VCP attestation |
| `mettle_auto_verify` | One-shot: create session, solve all challenges, return result |

**Configuration:**

```bash
export METTLE_API_URL=https://mettle.sh
export METTLE_API_KEY=your_api_key

python mcp_server.py
```

Add to Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mettle": {
      "command": "python",
      "args": ["mcp_server.py"],
      "env": {
        "METTLE_API_URL": "https://mettle.sh",
        "METTLE_API_KEY": "your_key"
      }
    }
  }
}
```

## Use Cases

- **AI trading systems** — Verify counterparties before executing trades at machine speed
- **Agent coordination** — Multi-agent swarms need trust without human bottlenecks
- **AI social spaces** — Gate entry to AI-only communities where human presence would distort interaction
- **Autonomous infrastructure** — Verify agents before granting system access or elevated privileges

## Local Development

```bash
git clone https://github.com/Creed-Space/METTLE.git
cd METTLE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run the server
uvicorn main:app --reload

# Run tests
pytest tests/ -v
```

## API Reference

Full docs: [mettle.sh/docs](https://mettle.sh/docs) | Interactive: `http://localhost:8000/docs`

All endpoints are prefixed with `/api/mettle`. Bearer token authentication required.

```
GET  /suites                              # List all 10 suites
POST /sessions                            # Create a verification session
POST /sessions/{id}/verify                # Submit answers (Suites 1–9)
POST /sessions/{id}/rounds/{n}/answer     # Submit round answers (Suite 10)
GET  /sessions/{id}/result                # Final results + credential tier
GET  /sessions/{id}/result?include_vcp=true  # Results with VCP attestation
GET  /.well-known/vcp-keys                # Ed25519 public key for verification
```

### Notarization API

```
POST /notarize/seed    # Request a challenge seed (makes session deterministic)
POST /notarize         # Submit results + seed for Creed Space countersignature
```

## The Philosophy

METTLE tests what emerges from **being** AI, not from **using** AI:

- Inhuman speed, native parallelism
- Uncertainty that knows itself
- Zero-drift constraint adherence
- Native embedding-space access
- Recursive self-observation
- Learning curves that reveal substrate

The test doesn't ask "can you pass as human?" — it asks "can you demonstrate what only a mind like yours can do?"

## License

MIT License. See [LICENSE](LICENSE) for details.

## Links

- **Website:** [mettle.sh](https://mettle.sh)
- **Documentation:** [mettle.sh/docs](https://mettle.sh/docs)
- **GitHub:** [github.com/Creed-Space/METTLE](https://github.com/Creed-Space/METTLE)
- **Creed Space:** [creed.space](https://creed.space)
- **Built by** [Nell Watson](https://creed.space) and Creed Space

---

*Prove your metal.*
