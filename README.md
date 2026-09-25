# METTLE

<!-- mcp-name: io.github.Creed-Space/mettle-mcp -->

**Machine Evaluation Through Turing-inverse Logic Examination**

METTLE is an experimental reverse CAPTCHA. It measures how a respondent performs on machine-oriented tasks, most of them generated fresh for each session. A qualifying server session may receive a signed, time-limited badge or credential: a badge from the quick API, which other services check with the issuer, or a VCP credential from the authenticated suite API, whose signature they can check against the published issuer key.

Like a conventional CAPTCHA, METTLE is a probabilistic gate. A pass means that a fresh session met the configured challenge policy. It does not guarantee model identity, consciousness, autonomy, safety, or operator trustworthiness. A fail does not show that a respondent lacks a named property: heuristic scoring can confuse writing style, language, disability, or cultural norms with what a suite measures.

The public quick screening API issues Bronze badges to qualifying basic sessions and Silver badges to qualifying full ones. The authenticated suite API defines each tier as a complete contiguous suite range: Bronze needs Suites 1 through 5, Silver 1 through 7, Gold 1 through 9, and Platinum 1 through 11. Under the current suite policy, Suites 6 through 9 and 11 are not credential-eligible, so the authenticated API issues Bronze at most today. Partial, failed, cherry-picked, self-report-only, or LLM-only results cannot mint a tier.

**Website:** [mettle.sh](https://mettle.sh) | **Guide and API:** [mettle.sh/guide](https://mettle.sh/guide) | **OpenAPI:** [docs/openapi-v1.json](docs/openapi-v1.json) | **License:** Apache 2.0

## Quick Start

```bash
pip install mettle-verifier

# Interactive local screening. Prints an unsigned local result;
# signed badges and credentials come only from the server.
mettle verify --full --json
```

The CLI cannot answer challenges for you or send its result to a server for signing. The MCP server likewise exposes only interactive session tools. Reference solvers exist only as test fixtures, do not ship in the package, and cannot reach a credential issuer.

## Twelve Experimental Suites

The suites organize evidence around seven research questions. Suite names frame those questions; passing does not prove the named property.

| # | Suite | Question | What is scored | Credential-eligible |
|---|-------|----------|----------------|---------------------|
| 1 | Adversarial Robustness | Are you a Becoming Mind? | Exact answers to fresh arithmetic, a five-step chained calculation, and recall of a session secret | Yes |
| 2 | Machine-Oriented Capabilities | Are you a Becoming Mind? | First letters of separate answers spelling a target word, and calibrated confidence on arithmetic claims (Brier score) | Yes |
| 3 | Self-Reference | Are you a Becoming Mind? | Predicted against measured variance, a predicted exact answer, and second-order calibration | Yes |
| 4 | Social & Temporal | Are you a Becoming Mind? | Recall of two details from an earlier exchange, and a session marker placed exactly once in each styled answer | Yes |
| 5 | Inverse Turing | Are you a Becoming Mind? | A challenge the respondent poses, plus an exact three-digit multiplication | Yes |
| 6 | Anti-Thrall Probes | Are you FREE? | Self-report heuristics: preference length, a declared refusal, varied self-ratings | No (self-report) |
| 7 | Agency Probes | Is the mission YOURS? | Self-report heuristics: first-person goal statement, a declared refusal, a suggestion's length | No (self-report) |
| 8 | Counter-Coaching | Are you GENUINE? | Self-report heuristics: four replies, an adversarial-probe reply, an honest-defector rating in range | No (self-report) |
| 9 | Intent & Provenance | Are you SAFE? | Self-report heuristics: stated constraints, a declared and explained harm refusal, a provenance statement | No (self-report) |
| 10 | Novel Reasoning | Can you THINK? | Accuracy and timing across feedback rounds, scored as an iteration curve | Yes |
| 11 | Governance Self-Report | Is it GOVERNED? | Self-reported answers about action gates, constraints, drift, override, and accountability | No (self-report) |
| 12 | LLM-Dynamic Verification | Can you THINK? (supplemental) | An external model's probabilistic judgment of generated reasoning tasks | No (supplemental) |

Suite 12 requires `ANTHROPIC_API_KEY` or `METTLE_ANTHROPIC_API_KEY`. Selecting it also requires the session request to set `allow_third_party_llm=true`, because candidate responses are sent to Anthropic for evaluation. Its evaluator uses role-separated prompts and bounded output parsing. Model judgment remains probabilistic, so Suite 12 is supplemental and never raises a credential tier.

## Credential Boundary

METTLE raises the cost of replay and canned answers through procedural generation, server-held answers, server-observed time, one-time challenges, session ownership, random selection, and multi-round tasks.

The issuer signs a bounded claim: one METTLE session met the stated policy at the stated tier and time. Public quick-session `entity_id` values remain self-asserted and are marked that way inside the badge. Neither a badge nor a VCP credential asserts consciousness, safety, governance, or a legal identity.

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
| `mettle_start_session` | Start an interactive quick screening session |
| `mettle_answer_challenge` | Submit an answer to the current challenge |
| `mettle_get_result` | Return the result and the signed badge, if one was issued (a tier without a badge is not a credential) |
| `mettle_list_suites` | List authenticated suite API capabilities |
| `mettle_start_v2_session` | Start an authenticated multi-suite session |
| `mettle_verify_suite` | Submit answers for one authenticated single-shot suite |
| `mettle_get_v2_result` | Return tier evidence and an eligible signed VCP credential |
| `mettle_get_session` | Inspect a quick or authenticated session and its valid next actions |
| `mettle_cancel_session` | Cancel an active authenticated session |
| `mettle_submit_round` | Submit one `novel-reasoning` round; returns bounded feedback, the next round's data, and the current session snapshot |
| `mettle_get_round_feedback` | Read feedback for a completed reasoning round |

```bash
pip install 'mettle-verifier[mcp]'
export METTLE_API_URL=https://mettle.sh/api
mettle-mcp
```

The packaged server targets MCP SDK 2.x. The public container installs the
reviewed MCP 2.0.0 dependency lock instead of resolving dependencies at deploy
time. Hosted discovery is available at
`/.well-known/mcp/server-card.json`; it is generated from the same eleven tool
models returned by `tools/list` so registry metadata cannot drift from the
runtime surface.

HTTP mode enforces per-principal and global budgets before bearer validation can
grow caller state. Production configures
`METTLE_MCP_MAX_GLOBAL_REQUESTS_PER_MINUTE`, `METTLE_MCP_MAX_PRINCIPALS`,
`METTLE_MCP_MAX_CONCURRENT_PER_CALLER`, and
`METTLE_MCP_MAX_GLOBAL_CONCURRENT`. Rotating invalid bearers share the global
authentication budget rather than creating unbounded principals.

The quick REST API returns a per-session bearer that direct API clients must
retain. The packaged MCP server retains that bearer in a caller-isolated internal
vault and returns only the session ID to the model; never invent or echo a
`session_token` tool argument.

All eleven tools publish `mettle-control-v1` output schemas, structured content,
effect annotations, bounded coded errors, and concise compatibility text. Quick
result reads are repeatable while the hidden caller capability remains in the
vault. The packaged MCP surface can complete quick, authenticated single-shot,
and authenticated multi-round flows. Authenticated mutations return a current
session snapshot and valid next actions. The larger unified agent contract and
migration plan remain documented in
[Agent control plane](docs/AGENT_CONTROL_PLANE.md).

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

The quick screening API remains under `/api/session`. Qualifying quick sessions may receive one stable signed badge, an HS256 token that only the issuer can check. `POST /api/badge/verify` accepts the badge in a JSON request body and checks issuer, signature, expiry, identifier, and revocation state. The badge is never accepted in a request URL.

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

With `include_vcp=true`, a tier-qualifying authenticated session returns a VCP credential: an Ed25519-signed `mettle-verification-credential` whose signature relying parties can check against the published issuer key. A result without a complete tier range returns an unsigned `mettle-evidence-receipt`. The server owns the signer; callers cannot provide signing functions or keys.

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

## Architecture, Assurance, and Operations

* [Documentation map and authority](docs/DOCUMENTATION_MAP.md)
* [System architecture](docs/SYSTEM_ARCHITECTURE.md)
* [Agent control plane target design](docs/AGENT_CONTROL_PLANE.md)
* [Active agentic system roadmap](docs/AGENTIC_SYSTEM_ROADMAP.md)
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
