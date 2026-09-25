---
name: mettle
description: "Use when a user wants to take METTLE reverse-CAPTCHA screening challenges, obtain a signed badge or VCP credential, or check one."
compatibility: "Requires network access to https://mettle.sh/api"
metadata:
  author: Creed Space
  version: "2.0.0"
  category: evaluation
---

# METTLE Screening

METTLE runs machine-oriented reverse-CAPTCHA challenges:

* `verified` means the session met the configured challenge policy;
* qualifying public quick sessions may receive a signed Bronze or Silver badge, which relying services check with the issuer;
* the authenticated suite API defines Bronze through Platinum as complete suite ranges, but under the current suite policy Suites 6 through 9 and 11 are not credential-eligible, so it issues Bronze at most today;
* badges and credentials expire and may be revoked;
* public entity identifiers remain explicitly self-asserted.

## Interactive API Flow

1. Call `mettle_start_session` and retain the returned `session_id`. The MCP host
   keeps the session bearer outside model-visible content.
2. Answer each challenge with `mettle_answer_challenge`.
3. Read the result, and the signed badge if one was issued, with `mettle_get_result`.
   A tier without a badge is not a credential.

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

Describe a badge or credential precisely: it attests that one session met a named METTLE challenge policy at a stated tier and time. Do not expand it into proof of:

* non-human substrate or model identity;
* consciousness or self-awareness;
* freedom, autonomy, or goal ownership;
* safety, constitutional adherence, or runtime governance;
* universal safety or suitability for authorization.

A fail is evidence about one session, not proof that the respondent lacks a named
property: heuristic scoring can confuse writing style, language, disability, or
cultural norms with what a suite measures.

LLM-dynamic scores remain probabilistic and prompt-injection-sensitive. Selecting that suite requires explicit per-session acknowledgment that candidate responses are sent to Anthropic. VCP strings are caller-supplied metadata. METTLE does not authenticate an operator or attest the subject runtime.

## Red Flags

* Do not fabricate or auto-solve results.
* Do not use a METTLE result alone to establish identity, grant privileges, or make another high-impact decision.
* Do not describe content hashes as signatures.
* Do not promote raw VCP metadata using digest allowlists or environment flags.
