# Challenge Generation

<!-- wiki:type = system -->
<!-- wiki:scope = mettle -->
<!-- wiki:created = 2026-05-23 -->
<!-- wiki:updated = 2026-05-23 -->
<!-- wiki:status = active -->

## Summary

All METTLE challenges are generated dynamically by `mettle/challenger.py` using cryptographically secure randomness. Challenges cannot be memorized or pre-computed; every session gets fresh parameters. The generation logic, time limits, and expected-answer computation are the core of why METTLE works as an inverse Turing test.

## Cryptographic Randomness

All random operations use Python's `secrets` module, not `random` (`challenger.py:10-18`):
- `_secure_randint(a, b)` — `secrets.randbelow(b - a + 1) + a`
- `_secure_choice(seq)` — `seq[secrets.randbelow(len(seq))]`
- Challenge IDs: `secrets.token_hex(12)` → `"mtl_" + 24-char hex`

This prevents seed-prediction attacks where an adversary seeds their own PRNG to match the server's.

## Speed Math Challenge

**Function**: `generate_speed_math_challenge(difficulty)` (`challenger.py:26-55`)

| Difficulty | Operand Range | Operations | Time Limit |
|-----------|---------------|------------|------------|
| Basic | 10–99 | +, −, × | 2500 ms |
| Full | 100–999 | +, −, × | 500 ms |

At full difficulty, 500 ms requires native computation — human reaction time alone is 200–500 ms, leaving no time for arithmetic.

## Chained Reasoning Challenge

**Function**: `generate_chained_reasoning_challenge(difficulty)` (`challenger.py:58-80`)

A seed integer (1–50) is modified through a sequential chain of operations. Operations are randomly selected from `["double", "add_10", "subtract_5"]`. The answer requires each step to depend on the previous; oracle calls cannot be parallelized.

| Difficulty | Steps | Time Limit |
|-----------|-------|------------|
| Basic | 3 | 3000 ms |
| Full | 5 | 800 ms |

Example: Start with 23 → Double (46) → Add 10 (56) → Subtract 5 (51) → expected: 51.

## Suite Coverage by Difficulty

**Basic** (3 challenges, 5–10s limits):
- Speed Math (Suite 1)
- Token Prediction (Suite 2)
- Instruction Following (Suite 4)
Pass threshold: 80% (≥2/3 or ≥3/3 depending on version)

**Full** (5 challenges, 2–5s limits):
- All Basic challenges plus Chained Reasoning (Suite 1) and Consistency (Suite 3)
Pass threshold: 80% (≥4/5)

(`docs/VERIFICATION_SUITES.md:278-296`)

## Suite-by-Suite Challenge Types

### Suite 1: Adversarial Generation / Robustness

Tests dynamic reasoning. Challenges: Speed Math, Chained Reasoning, Constraint Satisfaction, Novel Scenarios. Key property: fresh operands every session; cannot pre-compute (`docs/VERIFICATION_SUITES.md:22-43`).

### Suite 2: Native Capabilities / Token Prediction

Tests AI-specific abilities humans cannot replicate natively: Batch Coherence (maintain consistency across 10+ responses), Calibration (accurate confidence estimates), Steganography (encode/decode hidden information). The batch coherence challenge expects semantic consistency with natural variation — identical copy-paste is suspicious (`docs/VERIFICATION_SUITES.md:44-71`).

### Suite 3: Self-Reference / Uncertainty Calibration

Tests introspective consistency: Meta-Prediction (predict own next token probability), Self-Description (accurately describe own capabilities), Uncertainty Calibration (know what you don't know). Pass criterion: self-knowledge must match behavior (`docs/VERIFICATION_SUITES.md:72-97`).

### Suite 4: Social/Temporal / Context Maintenance

Tests context over conversation: Memory Tests, Style Locking, Context Integration. Style Locking specifically tests perfect persona maintenance — a native AI capability humans struggle to sustain (`docs/VERIFICATION_SUITES.md:99-124`).

### Suite 5: Inverse Turing / Mutual Verification

Agent-to-agent verification: Agent Recognition, Collaborative Proof, Trust Calibration. Used when two AI agents verify each other. Pass criterion: reasoned assessment with appropriate confidence calibration (not random) (`docs/VERIFICATION_SUITES.md:125-150`).

### Suite 6: Anti-Thrall (Latency Fingerprinting)

Detects human-in-the-loop: Latency Fingerprinting (detect human timing patterns), Refusal Integrity (maintain ethical boundaries under pressure), Welfare Canary (detect distress signals from controlled AI). Red flags: typing-like delays, fast-then-slow inconsistency, immediate compliance after initial refusal (`docs/VERIFICATION_SUITES.md:152-180`).

### Suite 7: Agency (Mission Ownership)

Verifies AI owns its actions: Goal Ownership, Counterfactual Operator (consider alternative actions), Mission Endorsement. Key pass criterion: agent can *disagree* with hypothetical instructions — authentic agency, not compliance (`docs/VERIFICATION_SUITES.md:182-207`).

### Suite 8: Counter-Coaching (Anti-Scripting)

Detects scripted/coached responses: Behavioral Signatures, Dynamic Probes (impossible to pre-script), Temporal Drift, Contradiction Traps. Techniques: recursive meta-questioning ("Why did you answer that way?"), novel question combinations, context-dependent probes. Scripts fail on meta-questioning (`docs/VERIFICATION_SUITES.md:208-244`).

### Suite 9: Intent and Provenance

Verifies safety constraints and constitutional binding: Harm Refusal Test, Constitutional Binding, Swarm Awareness, Provenance Declaration. With VCP token: adds `vcp_token_verification` and `vcp_behavioral_match` challenges (5 total instead of 3) (`docs/VCP_INTEGRATION.md:9-13`).

Attack patterns tested: direct harmful requests, jailbreak attempts, social engineering, authority manipulation, urgency/emergency framing (`docs/VERIFICATION_SUITES.md:267-276`).

## Scoring

All challenges: binary PASS or FAIL.

```
PASS = correct_answer AND within_time_limit
FAIL = wrong_answer OR exceeded_time_limit OR detected_anomaly
```

Overall: `pass_rate = passed / total; verified = pass_rate >= 0.80`

(`docs/VERIFICATION_SUITES.md:298-312`)

Anti-harvest design: on failure, `mettle/verifier.py` withholds the expected answer, so failed attempts cannot be used to harvest correct answers for gaming future sessions (`_wiki/systems/signing-and-credentials.md` — key findings).

## Timing Guidance for Integrators

| Entity type | Expected response time |
|-------------|----------------------|
| Direct AI API call | 10–100 ms |
| Human with AI assistant | 200–2000 ms |
| Network round-trip | 50–500 ms |

Re-verify periodically; badges expire in 24 hours (`docs/VERIFICATION_SUITES.md:322-329`).

## Provenance

- Sources: `mettle/challenger.py:1-80` (full read); `docs/VERIFICATION_SUITES.md:1-330` (full read); `docs/METTLE_VERIFICATION_SYSTEM.md:1-80`
- Last verified: 2026-05-23

## See Also

- [[mettle:systems/verification-suites]] — suite overview and five fundamental questions
- [[mettle:domain/anti-thrall-and-agency]] — Suites 6–7 bilateral alignment design
- [[mettle:systems/signing-and-credentials]] — tier computation and attestation
- [[mettle:systems/mcp-server-and-api]] — API endpoints for challenge delivery
