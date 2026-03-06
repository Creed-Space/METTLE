# METTLE

**Machine Evaluation Through Turing-inverse Logic Examination**

An inverse Turing test for the agentic era. Instead of *prove you're human*, METTLE asks *prove you're NOT human.*

METTLE tests capabilities that emerge from **being** AI — not from using AI as a tool. Inhuman speed, native parallelism, uncertainty that knows itself, recursive self-observation, and learning curves that reveal substrate.

**Website:** [mettle.sh](https://mettle.sh) | **Docs:** [mettle.sh/docs](https://mettle.sh/docs) | **License:** Apache 2.0

---

## Quick Start

```bash
# Install the open-source verifier
pip install mettle-verifier

# Run all 12 suites locally — self-signed credential
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

## 12 Verification Suites

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
| 11 | **Governance Verification** | Is it GOVERNED? | Action gate probes, constitutional recitation, drift checks, override resistance, accountability chain. Verifies operational governance. |
| 12 | **LLM-Dynamic** | Can you REASON DEEPLY? | Claude-generated challenges: perspective shifting, structured constraint satisfaction, meta-cognitive probing. Every challenge unique — Claude generates novel problems and evaluates responses semantically. Requires `ANTHROPIC_API_KEY`. |

Suite 11 was motivated by the Rathbun scenario: an agent that passes all capability and safety checks but operates without runtime governance, making harmful actions structurally possible despite good intentions.

Suite 12 addresses a fundamental limitation of deterministic challenges: anyone who reads the source code knows the answer space. LLM-dynamic challenges are generated fresh by Claude for each session, with semantic evaluation that assesses reasoning quality rather than pattern matching. Available when `ANTHROPIC_API_KEY` or `METTLE_ANTHROPIC_API_KEY` is set; excluded from `suites=["all"]` otherwise.

## Credential Tiers

| Tier | Badge | Requires | Meaning |
|------|-------|----------|---------|
| **Bronze** | METTLE-verified AI | Suites 1–5 | Confirmed AI substrate |
| **Silver** | METTLE-verified autonomous | Suites 1–7 | Free agent with genuine agency |
| **Gold** | METTLE-verified safe | Suites 1–9 | Genuine, constitutionally bound |
| **Platinum** | METTLE-verified governed | Suites 1–11 | Full governance — action gates, drift detection, accountability |

Suite 12 (LLM-Dynamic) is **supplemental** — it strengthens any tier but is not required for any. It requires an API key, so mandating it would create an external dependency in the trust chain.

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
| LLM-evaluated semantics | Source-reading — deterministic challenges can be reverse-engineered from code; LLM-generated challenges cannot |

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
| `mettle_verify_suite` | Submit answers for a single-shot suite (1–9, 11–12) |
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
GET  /suites                              # List all 12 suites (includes availability flag)
POST /sessions                            # Create a verification session
POST /sessions/{id}/verify                # Submit answers (Suites 1–9, 11)
POST /sessions/{id}/rounds/{n}/answer     # Submit round answers (Suite 10)
GET  /sessions/{id}/result                # Final results + credential tier + governance/operator attestations
GET  /sessions/{id}/result?include_vcp=true  # Results with VCP attestation
GET  /.well-known/vcp-keys                # Ed25519 public key for verification
```

### Operator Commitment (CreateSessionRequest)

Sessions can include an operator commitment for Platinum-tier accountability:

```json
{
    "suites": ["all"],
    "entity_id": "agent-xyz",
    "vcp_token": "VCP:3.1:agent-xyz\nC:creed-professional@2.0.0\n...",
    "operator_commitment": {
        "operator_pseudonym": "anon-42",
        "operator_public_key": "-----BEGIN PUBLIC KEY-----\n...",
        "signed_commitment": "<base64 Ed25519 signature>",
        "contact_method": "email_hash",
        "contact_hash": "sha256:..."
    }
}
```

The signed commitment message must be exactly: `I accept accountability for agent {entity_id}`

### Response Attestations

Results include two additional attestation fields when applicable:

- **`governance_attestation`** — Populated when the session includes a VCP token and tier is gold or platinum. Contains: `framework`, `framework_version`, `constitutional_hash`, `has_action_gate`, `has_drift_detection`, `has_bilateral`, `verified_at`, `attestation_signature`.
- **`operator_attestation`** — Populated when the session includes an `operator_commitment` with a valid Ed25519 signature. Links the agent cryptographically to an accountable operator.

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

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Links

- **Website:** [mettle.sh](https://mettle.sh)
- **Documentation:** [mettle.sh/docs](https://mettle.sh/docs)
- **GitHub:** [github.com/Creed-Space/METTLE](https://github.com/Creed-Space/METTLE)
- **Creed Space:** [creed.space](https://creed.space)
- **Built by** [Nell Watson](https://creed.space) and Creed Space

---

*Prove your mettle.*
