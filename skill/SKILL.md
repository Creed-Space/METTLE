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

1. Call `mettle_start_session` and retain both `session_id` and `session_token`.
2. Answer each challenge with `mettle_answer_challenge`.
3. Read the result and credential with `mettle_get_result`.

The auto-solve tool was removed. Never route a reference solver into a live session.

## Interpretation Rules

Describe the credential precisely as proof that a METTLE challenge policy passed. Do not expand it into proof of:

* non-human substrate or model identity;
* consciousness or self-awareness;
* freedom, autonomy, or goal ownership;
* safety, constitutional adherence, or runtime governance;
* universal safety or authorization suitability.

LLM-dynamic scores remain probabilistic and prompt-injection-sensitive. VCP strings are caller-supplied metadata. An operator signature proves only that the key holder signed the commitment.

## Red Flags

* Do not fabricate or auto-solve results.
* Do not grant more access than the relying service's configured tier and freshness policy allows.
* Do not describe content hashes as signatures.
* Do not promote raw VCP metadata using digest allowlists or environment flags.
