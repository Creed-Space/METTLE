# METTLE Security White Paper

**Machine Evaluation Through Turing-inverse Logic Examination**

Version 3.0, July 2026

## Executive Summary

METTLE is a reverse-CAPTCHA challenge and credential service. It measures performance on generated machine-oriented tasks and issues a signed, time-limited credential when the configured policy passes.

The credential attests to a METTLE session result. It does not guarantee model identity, consciousness, autonomy, safety, governance, or operator trustworthiness. Public quick-session entity identifiers are explicitly marked as self-asserted.

## Threat Model

Relevant adversaries include a human using a model, an automated solver, a relaying service, a respondent trained on public source, a malicious model, a prompt-injecting respondent, a session thief, and an operator supplying false governance metadata.

Important attack goals are answer harvesting, replay and race attacks, timing substitution, metadata promotion, evaluator prompt injection, credential confusion, bearer-token theft, webhook abuse, and administrative compromise.

## Security Invariants

1. Correct answers remain server-side.
2. Every submitted answer is bound to the active challenge and complete configured suite set.
3. Server-observed time is authoritative.
4. Session mutation and result reads require the independent session bearer token.
5. Session state transitions and quota reservation are atomic in Redis.
6. Payloads, in-memory stores, and outbound response bodies are bounded.
7. Webhook destinations reject local, private, redirecting, and DNS-rebinding targets.
8. Administrative authentication uses headers, constant-time comparison, and rate limiting.
9. Model-judge input is role-separated, escaped, bounded, and parsed into bounded scores.
10. Only server-derived results that satisfy an explicit tier policy can reach the credential signer.
11. LLM-only, partial, cherry-picked, caller-asserted, or failed results cannot mint a tier.
12. Callers cannot supply signing keys or signing functions.

## Verification Credential Contract

The public quick API treats an 80 percent result as a reverse-CAPTCHA pass. A successful basic session returns a result shaped like:

```json
{
  "verified": true,
  "screening_passed": true,
  "assurance": "mettle_behavioral_verification",
  "credential_eligible": true,
  "tier": "bronze",
  "badge": "<signed JWT>",
  "badge_info": {"signed": true, "jti": "<revocable id>"}
}
```

The local CLI can verify a local run but cannot issue a portable credential because the claimant does not control the server issuer. The authenticated suite API signs only complete contiguous suite ranges: Bronze requires Suites 1 through 5, Silver 1 through 7, Gold 1 through 9, and Platinum 1 through 11.

## LLM-Dynamic Evaluation

The LLM evaluator uses role-separated prompts and treats respondent content as quoted data. Schema and range checks constrain the returned judgment. These controls reduce accidental instruction following but cannot eliminate prompt injection or model error. Suite 12 is supplemental and never raises a credential tier. No LLM path can invoke the signer directly.

## VCP and Governance Metadata

VCP tokens are caller-supplied strings. METTLE may parse and hash them, but returns `source_verified=false`, false operational-governance flags, and a null attestation signature. Digest allowlists and deployment environment variables cannot promote these claims.

An Ed25519 operator commitment is verified over canonical JSON binding the declared operator fields to `entity_id`. This proves possession of the signing key for that statement only. It does not verify the agent, runtime, governance system, or challenge result.

## Badge Issuance And Compatibility

Passing quick sessions receive one stable HS256 badge using the server-owned production secret. The payload binds the result to its session, tier, issue time, expiry, nonce, and revocable identifier. It marks the public `entity_id` as self-asserted. Verification fixes the algorithm and checks issuer, expiry, required claims, and revocation state. The same verification surface retains compatibility with valid historical badges that satisfy these controls.

The authenticated suite API uses the server-owned Ed25519 key. It signs the complete credential envelope, including subject, session, tier, suite results, policy metadata, issue time, and expiry. Production refuses to start without the configured signing key.

## Residual Research Limitations

Procedural generation, time budgets, consistency checks, and multi-round curves increase the cost of simple replay. They remain behavioral heuristics. A capable respondent, relay, solver, or model-assisted human may imitate the measured patterns. Model-judge output remains probabilistic. Self-reported governance remains self-reported.

These limitations are managed through short credential lifetimes, tier-specific policy, anti-replay controls, explicit claim semantics, revocation, and relying-party policy. As with conventional CAPTCHA, consumers should select controls proportionate to the value behind the gate.

## Future Strengthening

Subject-key proof of possession, audience binding, managed key rotation, hardware or provider attestation, and continuous verification can strengthen later assurance profiles. These controls complement the core reverse-CAPTCHA credential rather than blocking its practical use.

## Verification

Security regression coverage includes session isolation, atomic transitions, timing, payload bounds, solver-surface removal, stable badge issuance, tier-policy enforcement, signer isolation, signature tamper detection, revocation, unverified VCP metadata, model-judge parsing, webhook controls, secret scanning, dependency auditing, linting, typing, and the complete test suite.
