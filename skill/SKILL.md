---
name: mettle
description: "Use when a user wants to complete METTLE reverse-CAPTCHA challenges, obtain a signed result, or verify a METTLE credential."
compatibility: "Requires network access to https://mettle.sh/api"
metadata:
  author: Creed Space
  version: "2.0.0"
  category: evaluation
---

# METTLE Verification

METTLE runs machine-oriented reverse-CAPTCHA challenges:

* `verified` means the configured challenge policy passed.
* passing public quick sessions receive a signed Bronze or Silver badge;
* authenticated suite sessions can earn Bronze through Platinum by completing every required suite in the tier range;
* credentials expire and may be revoked;
* public entity identifiers remain explicitly self-asserted.

## Interactive API Flow

1. Call `mettle_start_session` and retain the returned `session_id`. The MCP host
   keeps the session bearer outside model-visible content.
2. Answer each challenge with `mettle_answer_challenge`.
3. Read the result and credential with `mettle_get_result`.

The auto-solve tool was removed. Never route a reference solver into a live session.

Do not supply or request a `session_token` tool argument. Direct REST clients have a
different responsibility and must retain the bearer returned by the API.

## Mode Selection and Current Limits

Use the three quick tools above for a complete Bronze or Silver screening flow.
Use `mettle_list_suites`, `mettle_start_v2_session`, `mettle_verify_suite`, and
`mettle_get_v2_result` for authenticated single-shot suites. Use
`mettle_submit_round` and `mettle_get_round_feedback` for `novel-reasoning`.
Use `mettle_get_session` to inspect either profile and
`mettle_cancel_session` to cancel an authenticated session.

If a tool reports an unknown or expired quick session, the host no longer has the
hidden bearer. Do not guess it. Start a new session only if consuming another
session and quota is acceptable.

Quick result reads are repeatable while the hidden caller capability remains in
the MCP vault. Prefer structured content over parsing compatibility text. Follow
the returned actions rather than guessing call order.

## Interpretation Rules

Describe the credential precisely as proof that a METTLE challenge policy passed. Do not expand it into proof of:

* non-human substrate or model identity;
* consciousness or self-awareness;
* freedom, autonomy, or goal ownership;
* safety, constitutional adherence, or runtime governance;
* universal safety or authorization suitability.

LLM-dynamic scores remain probabilistic and prompt-injection-sensitive. Selecting that suite requires explicit per-session acknowledgement that candidate responses are sent to Anthropic. VCP strings are caller-supplied metadata. METTLE does not authenticate an operator or attest the subject runtime.

## Red Flags

* Do not fabricate or auto-solve results.
* Do not use a METTLE result alone to establish identity, grant privileges, or make another high-impact decision.
* Do not describe content hashes as signatures.
* Do not promote raw VCP metadata using digest allowlists or environment flags.
