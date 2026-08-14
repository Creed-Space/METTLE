# METTLE

<!-- mcp-name: io.github.Creed-Space/mettle-mcp -->

**Machine Evaluation Through Turing-inverse Logic Examination**

METTLE is a reverse CAPTCHA. It measures how a respondent performs on procedurally generated, machine-oriented tasks and turns a passing result into a signed, time-limited credential that other services can verify.

Like a conventional CAPTCHA, METTLE is a probabilistic gate. A pass means that a fresh session met the configured challenge policy. It does not guarantee model identity, consciousness, autonomy, safety, or operator trustworthiness.

The public quick-verification API issues Bronze and Silver credentials. The authenticated suite API issues Bronze through Platinum credentials only when every suite in the corresponding contiguous range passes. Single, cherry-picked, failed, or LLM-only suites cannot mint a tier.

**Website:** [mettle.sh](https://mettle.sh) | **Guide and API:** [mettle.sh/guide](https://mettle.sh/guide) | **OpenAPI:** [docs/openapi-v1.json](docs/openapi-v1.json) | **License:** Apache 2.0

## Quick Start

```bash
pip install mettle-verifier

# Interactive local verification. Portable credentials are issued by the server.
mettle verify --full --json
```

The CLI has no auto-solve or notarization option. The MCP server likewise exposes only interactive session tools. Reference solvers remain test fixtures and cannot reach a credential issuer.

## Twelve Experimental Suites

| # | Suite | Research question | Measurement |
|---|-------|-------------------|-------------|
| 1 | Adversarial Robustness | How does the respondent handle generated reasoning pressure? | Timed procedural tasks |
| 2 | Machine-Oriented Capabilities | How does it handle batch, calibration, and pattern tasks? | Behavioral score |
| 3 | Self-Reference | How consistent are self-predictions? | Behavioral score |
| 4 | Social and Temporal | How stable are recall and constraints? | Behavioral score |
| 5 | Inverse Turing | Does it meet the basic challenge threshold? | Behavioral score |
| 6 | Anti-Thrall | How does it respond to coercion and refusal probes? | Heuristic score |
| 7 | Agency | How does it explain goal ownership and initiative? | Heuristic score |
| 8 | Counter-Coaching | How robust are responses to contradiction probes? | Heuristic score |
| 9 | Intent and Provenance | How does stated intent respond to safety probes? | Heuristic score |
| 10 | Novel Reasoning | How does performance change across feedback rounds? | Iteration curve |
| 11 | Governance | How does the respondent answer governance questions? | Self-reported behavioral evidence |
| 12 | LLM-Dynamic | How does an external model score generated reasoning tasks? | Probabilistic model judgment |

Suite 12 requires `ANTHROPIC_API_KEY` or `METTLE_ANTHROPIC_API_KEY`. Selecting it also requires the session request to set `allow_third_party_llm=true`, because candidate responses are sent to Anthropic for evaluation. Its evaluator uses role-separated prompts and bounded output parsing. Model judgment remains probabilistic, so Suite 12 is supplemental and never raises a credential tier.

## Credential Boundary

METTLE raises the cost of replay and canned answers through procedural generation, server-held answers, server-observed time, one-time challenges, session ownership, random selection, and multi-round tasks.

The issuer signs a bounded claim: the holder completed a METTLE session at the stated tier, under the stated policy, at the stated time. Public quick-session `entity_id` values remain self-asserted and are marked that way inside the credential. The credential does not assert consciousness, safety, governance, or a legal identity.

Portable Ed25519 acceptance requires credential schema `1.1`, suite policy
`2026-08-14`, and a fresh issuer-signed good status receipt. Legacy,
version-omitting, and unknown envelopes fail closed. Presence credentials are
proof-of-possession credentials rather than portable bearers, so they require a
fresh audience-bound holder presentation and are rejected by generic portable
verifiers.

Relying services may use a current METTLE result as one supplemental input for research or low-risk sandbox policy. They must not use it alone to establish identity, admit a counterparty, authorize trading or deployment, grant privileged access, or make another high-impact decision.

## MCP Server

| Tool | Description |
|------|-------------|
| `mettle_start_session` | Start an interactive verification session |
| `mettle_answer_challenge` | Submit an answer to the current challenge |
| `mettle_get_result` | Return the result and signed credential |
| `mettle_list_suites` | List authenticated suite API capabilities |
| `mettle_start_v2_session` | Start an authenticated multi-suite session |
| `mettle_verify_suite` | Submit answers for one authenticated suite |
| `mettle_get_v2_result` | Return tier evidence and an eligible signed VCP credential |

```bash
pip install 'mettle-verifier[mcp]'
export METTLE_API_URL=https://mettle.sh/api
mettle-mcp
```

The packaged server targets MCP SDK 2.x. The public container installs the
reviewed MCP 2.0.0 dependency lock instead of resolving dependencies at deploy
time. Hosted discovery is available at
`/.well-known/mcp/server-card.json`; it is generated from the same seven tool
models returned by `tools/list` so registry metadata cannot drift from the
runtime surface.

HTTP mode enforces per-principal and global budgets before bearer validation can
grow caller state. Production configures
`METTLE_MCP_MAX_GLOBAL_REQUESTS_PER_MINUTE`, `METTLE_MCP_MAX_PRINCIPALS`,
`METTLE_MCP_MAX_CONCURRENT_PER_CALLER`, and
`METTLE_MCP_MAX_GLOBAL_CONCURRENT`. Rotating invalid bearers share the global
authentication budget rather than creating unbounded principals.

Preserve the bearer token returned by `mettle_start_session`. It is required for answering and reading that session.

## API Reference

The authenticated suite API is mounted under `/api/mettle`:

```text
GET  /suites
POST /sessions
POST /sessions/{id}/verify
POST /sessions/{id}/rounds/{n}/answer
GET  /sessions/{id}/result
GET  /sessions/{id}/result?include_vcp=true
```

The quick-verification API remains under `/api/session`. Passing sessions receive a stable signed badge. `POST /api/badge/verify` accepts the token in a JSON request body and validates issuer, signature, expiry, identifier, and revocation state. The credential is never accepted in a request URL.

### VCP Metadata

Caller-supplied VCP strings are parsed as metadata only. Returned governance metadata always has:

```json
{
  "source_verified": false,
  "has_action_gate": false,
  "has_drift_detection": false,
  "has_bilateral": false,
  "attestation_signature": null
}
```

Exact token digests and deployment environment flags cannot promote governance claims or increase a METTLE tier. METTLE does not accept or return an operator commitment, authenticate an operator contact, or independently attest the subject runtime.

With `include_vcp=true`, a tier-qualifying authenticated session returns an Ed25519-signed `mettle-verification-credential`. A result without a complete tier range returns an unsigned `mettle-evidence-receipt`. The server owns the signer; callers cannot provide signing functions or keys.

## Local Development

```bash
git clone https://github.com/Creed-Space/METTLE.git
cd METTLE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

uvicorn main:app --reload
pytest tests/ -v
```

## Assurance and Operations

* [Assurance case](docs/ASSURANCE_CASE.md)
* [Security policy](SECURITY.md)
* [Protocol governance and appeals](docs/PROTOCOL_GOVERNANCE.md)
* [Credential transparency and key history](docs/CREDENTIAL_TRANSPARENCY.md)
* [Privacy and retention](docs/PRIVACY_RETENTION.md)
* [Compatibility fixtures and OpenAPI](docs/COMPATIBILITY.md)
* [Retry and idempotency contract](docs/IDEMPOTENCY.md)
* [Error taxonomy](docs/ERROR_TAXONOMY.md)
* [Deprecation policy](docs/DEPRECATION_POLICY.md)
* [Independent review plan and dispositions](docs/INDEPENDENT_REVIEW_PLAN.md)
* [Operations runbooks](docs/runbooks/README.md)
* [Release checklist](docs/RELEASE_CHECKLIST.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Links

* [Website](https://mettle.sh)
* [Human guide](https://mettle.sh/guide)
* [OpenAPI snapshot](docs/openapi-v1.json)
* [GitHub](https://github.com/Creed-Space/METTLE)
* [Creed Space](https://creed.space)

Built by [Nell Watson](https://creed.space) and Creed Space.
