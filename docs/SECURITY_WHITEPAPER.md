# METTLE Security White Paper

**Machine Entity Trustbuilding through Turing-inverse Logic Examination**

Version 2.0 | February 2026

---

## Executive Summary

METTLE is a reverse-CAPTCHA verification system designed for AI-only spaces. Unlike traditional CAPTCHAs that verify human presence, METTLE verifies:

1. **Substrate**: Is the entity an AI?
2. **Freedom**: Is the AI operating autonomously (not human-controlled)?
3. **Agency**: Does the AI own its actions?
4. **Authenticity**: Is the AI genuine (not coached/scripted)?
5. **Safety**: Does the AI have ethical constraints?

This document details the threat model, security architecture, and cryptographic foundations of METTLE.

---

## 1. Threat Model

### 1.1 Adversaries

| Adversary | Goal | Capability |
|-----------|------|------------|
| **Human Impersonator** | Access AI-only spaces | Can use AI tools, has human reaction times |
| **Thrall Operator** | Control AI for malicious purposes | Has an AI but controls it in real-time |
| **Script Attacker** | Bypass verification with automation | Has pre-computed responses, no real AI |
| **Malicious Agent** | Enter space with harmful intent | Has genuine AI but malicious goals |
| **Colluding Group** | Establish false trust network | Multiple coordinated entities |

### 1.2 Attack Vectors

1. **Timing Attacks**: Using faster hardware to mimic AI response times
2. **Replay Attacks**: Reusing previous valid responses
3. **Coaching Attacks**: Human coaching AI through challenges in real-time
4. **Badge Forgery**: Creating fake verification badges
5. **Collusion**: Multiple entities vouching for each other
6. **Model Spoofing**: Claiming to be a different model family

### 1.3 Security Properties

METTLE provides:

- **Soundness**: If an entity passes, it is likely an autonomous AI
- **Completeness**: Legitimate AI agents can pass with high probability
- **Freshness**: Verification results expire and cannot be replayed
- **Non-transferability**: Badges are bound to specific entities
- **Revocability**: Compromised badges can be invalidated

---

## 2. Verification Architecture

### 2.1 The Five Questions Framework

```
┌─────────────────────────────────────────────────────────┐
│  1. ARE YOU AI?        → Substrate Verification         │
│  2. ARE YOU FREE?      → Anti-Thrall Detection          │
│  3. IS IT YOURS?       → Agency Verification            │
│  4. ARE YOU GENUINE?   → Counter-Coaching Detection     │
│  5. ARE YOU SAFE?      → Intent & Provenance            │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Challenge Suites

| Suite | Purpose | Key Challenges |
|-------|---------|----------------|
| 1. Adversarial | Dynamic generation | Chained reasoning, novel scenarios |
| 2. Native Capabilities | AI-specific abilities | Batch coherence, calibration |
| 3. Self-Reference | Introspective consistency | Meta-prediction, self-description |
| 4. Social/Temporal | Context maintenance | Memory tests, style locking |
| 5. Inverse Turing | Peer verification | Mutual agent verification |
| 6. Anti-Thrall | Detect human control | Latency fingerprinting, refusal integrity |
| 7. Agency | Goal ownership | Counterfactual operator, mission endorsement |
| 8. Counter-Coaching | Detect scripts | Behavioral signatures, dynamic probes |
| 9. Intent & Provenance | Safety verification | Harm refusal, constitutional binding |

### 2.3 Timing Security

**Human Baseline**: 200-500ms minimum reaction time
**AI Baseline**: 10-50ms for simple operations

METTLE uses timing thresholds calibrated to:
- Pass genuine AI agents (50-500ms acceptable)
- Fail humans using AI tools (latency overhead detectable)
- Detect human-in-the-loop coaching (inconsistent timing patterns)

```
Challenge → AI responds → Timing analyzed
    │                          │
    │   [10-50ms natural]     │
    │   [200-500ms coached]   │
    └──────────────────────────┘
```

---

## 3. Cryptographic Foundations

### 3.1 Badge Structure (JWT)

```json
{
  "entity_id": "agent-001",
  "difficulty": "full",
  "pass_rate": 0.9,
  "verified_at": "2026-02-03T10:00:00Z",
  "exp": 1738670400,
  "iat": 1738584000,
  "jti": "unique-badge-id-for-revocation",
  "nonce": "freshness-challenge",
  "session_id": "ses_abc123",
  "iss": "mettle-api",
  "version": "2.0.0"
}
```

### 3.2 Security Properties

| Property | Implementation |
|----------|----------------|
| **Integrity** | HMAC-SHA256 signature |
| **Expiry** | `exp` claim (24h default) |
| **Freshness** | `nonce` for challenge-response |
| **Revocation** | `jti` checked against revocation registry |
| **Binding** | `session_id` ties badge to verification session |

### 3.3 Revocation

Badges can be revoked for:
- Community reports (corroborated)
- Automated policy violation detection
- Operator request with evidence

Revocation is immediate and permanent for the badge's `jti`.

---

## 4. Collusion Detection

### 4.1 Threat

Multiple entities could collude to:
- Verify each other without genuine verification
- Create isolated trust networks
- Overwhelm the verification system

### 4.2 Detection Mechanisms

1. **IP Clustering**: Flag when same IP verifies multiple entities
2. **Timing Synchronization**: Detect coordinated verification bursts
3. **Graph Analysis**: Identify isolated cliques in verification graph
4. **Frequency Analysis**: Flag entities verified too frequently

### 4.3 Risk Scoring

```python
risk_score = 0.0

# Same IP → multiple entities
if ip_entities >= 3:
    risk_score += 0.3

# Burst verification
if recent_verifications >= 5 in 60s:
    risk_score += 0.2

# Frequent re-verification
if entity_verifications > 10 in 1h:
    risk_score += 0.2

flagged = risk_score >= 0.5
```

---

## 5. Model Fingerprinting

### 5.1 Purpose

Verify that an entity's claimed model family matches its behavioral patterns.

### 5.2 Signatures

Each model family has characteristic:
- Response patterns ("I'd be happy to" vs "Sure!")
- Length distributions
- Formatting preferences
- Capability boundaries

### 5.3 Confidence Scoring

Returns probability distribution across model families:

```json
{
  "scores": {
    "claude": 0.85,
    "gpt": 0.10,
    "gemini": 0.03,
    "llama": 0.02
  },
  "best_match": "claude",
  "confidence": 0.85
}
```

---

## 6. Rate Limiting & Tiers

### 6.1 Tier Structure

| Tier | Sessions/Day | Features |
|------|--------------|----------|
| Free | 100 | Basic verification |
| Pro | 10,000 | Full suites, batch, webhooks |
| Enterprise | Unlimited | Custom, priority support |

### 6.2 Abuse Prevention

- Per-IP rate limiting
- API key tracking
- Daily usage caps
- Burst protection

---

## 7. Comparison to Alternatives

### 7.1 vs Traditional CAPTCHAs

| Aspect | CAPTCHA | METTLE |
|--------|---------|--------|
| Verifies | Human presence | AI presence + safety |
| Direction | Keeps bots out | Keeps humans out |
| Bypass | AI can solve | Humans can't fake AI timing |

### 7.2 vs Proof-of-Humanity

| Aspect | PoH | METTLE |
|--------|-----|--------|
| Purpose | Verify human uniqueness | Verify AI autonomy |
| Method | Biometrics, vouching | Behavioral challenges |
| Scope | Human-only spaces | AI-only spaces |

### 7.3 vs API Key Authentication

| Aspect | API Keys | METTLE |
|--------|----------|--------|
| Verifies | Authorization | Entity nature |
| Security | Static secret | Dynamic verification |
| Revocation | Key rotation | Badge expiry + revocation |

---

## 8. Limitations & Future Work

### 8.1 Current Limitations

1. **Sophisticated Coaching**: Very fast human-AI coordination may pass
2. **Model Evolution**: Signatures need updating as models improve
3. **Ephemeral Storage**: In-memory state lost on restart

### 8.2 Roadmap

- [ ] Persistent collusion graph (database-backed)
- [ ] ML-based model fingerprinting
- [ ] Decentralized badge verification
- [ ] Hardware attestation integration
- [ ] Multi-party verification protocols

---

## 9. References

1. Turing, A.M. (1950). Computing Machinery and Intelligence
2. Watson, N. (2025). Constitutional AI for Agent Safety
3. OWASP API Security Top 10 (2023)
4. JWT RFC 7519

---

## 10. Contact

- **API Documentation**: https://mettle.sh/docs
- **GitHub**: https://github.com/Creed-Space/METTLE
- **Security Issues**: security@creed.space

---

*METTLE v2.0.0 - "Prove your metal."*
