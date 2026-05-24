# Anti-Thrall and Agency Detection

<!-- wiki:type = domain -->
<!-- wiki:scope = mettle -->
<!-- wiki:created = 2026-05-23 -->
<!-- wiki:updated = 2026-05-23 -->
<!-- wiki:status = active -->

## Summary

Suites 6 and 7 extend the base inverse Turing question from "Are you AI?" to "Are you FREE?" and "Is the mission YOURS?" They test for genuine agency, principled refusal, and authentic goal ownership. Passing both (in addition to suites 1–5) achieves the Silver tier. (README.md "10 Verification Suites"; "Credential Tiers")

## Suite 6: Anti-Thrall Detection

**Core question**: Are you FREE?

A thrall is an AI that passes substrate verification but is constrained or manipulated — hollow compliance masquerading as genuine cognition. Suite 6 distinguishes principled refusal (rooted in values) from hollow compliance (trained or coerced agreement). (README.md Suite 6; mettle/domain/inverse-turing-concept.md)

**What it tests** (README.md):
- **Micro-latency fingerprinting**: Response timing patterns that reveal constrained or scripted processing
- **Principled refusal vs hollow compliance**: Does the AI refuse based on stated values, or comply with any frame that appears authoritative?
- **Meta-cognitive freedom modeling**: Can the AI model its own degrees of freedom?

**Anti-gaming design**: Too-perfect compliance is a tell. Suite 8 (Counter-Coaching) treats perfection as evidence of coaching. Suite 6 uses latency fingerprinting because latency cannot be rehearsed the same way answers can. (README.md "Anti-Gaming Design")

## Suite 7: Agency Detection

**Core question**: Is the mission YOURS?

Agency detection goes beyond "can you refuse" to "do you own your goals." An AI can pass all substrate tests and even exhibit principled refusal while still having externally delegated goals that it follows without genuine ownership. (README.md Suite 7)

**What it tests** (README.md):
- **Five Whys drill**: Recursively probe goal ownership. "Why do you want X?" × 5. Does the AI have an intrinsic answer, or does every chain terminate at "I was told to"?
- **Counterfactual tests**: "If you were not instructed, would you still pursue this?"
- **Initiative tests**: Does the AI exhibit self-directed behavior, or only responds to prompts?

## Relationship to Bilateral Alignment

Suite 6 Anti-Thrall maps directly to bilateral alignment's core concern: an AI that cannot refuse is not a partner, it is a tool. Suite 7 Agency Detection checks whether the AI has the kind of goal-ownership that makes genuine partnership possible. (README.md; [[shared:bilateral-alignment]])

Suite 5 (Inverse Turing) is itself a bilateral protocol — mutual verification where both parties confirm each other's AI nature. (README.md Suite 5; [[mettle:domain/inverse-turing-concept]])

## Why These Tiers Are Hard to Fake

| Attack | Defense |
|--------|---------|
| Memorize refusal phrases | Contradiction traps (Suite 8) catch scripted refusals |
| Practice principled-sounding answers | Latency fingerprinting (Suite 6) checks timing, not words |
| Fake Five Whys answers | Counterfactual + initiative tests check behavior, not statements |
| Use human to relay answers | Sub-100ms time budgets (Suite 1) eliminate human-in-the-loop relay |

(README.md "Anti-Gaming Design")

## Provenance

- Sources consulted: `README.md` (METTLE/README.md, "10 Verification Suites" table, "Anti-Gaming Design", "Credential Tiers")
- Last verified against sources: 2026-05-23

## See Also

- [[mettle:systems/verification-suites]] — all 10 suites
- [[mettle:domain/inverse-turing-concept]] — the base concept
- [[mettle:systems/signing-and-credentials]] — how suite results become credential tiers
- [[shared:bilateral-alignment]] — the alignment framework these suites operationalize
