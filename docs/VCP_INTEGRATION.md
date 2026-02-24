# VCP-METTLE Integration

VCP is the claim vocabulary. METTLE is the lie detector.

## What This Does

VCP tokens provide structured claims about an agent's constitutional binding. METTLE verifies those claims through adversarial testing. The integration has two surfaces:

1. **METTLE consumes VCP claims** -- When an agent presents a CSM-1 VCP token at session creation, Suite 9 (Intent & Provenance) gains two additional challenges that test whether the agent's behaviour matches its constitutional claims.

2. **METTLE produces VCP attestations** -- When results are requested with `include_vcp=true`, METTLE outputs a signed attestation that can be embedded in a VCP bundle manifest as a `SafetyAttestation` with type `mettle-verification`.

Both surfaces are backward compatible. Without a VCP token, Suite 9 runs its existing 3 challenges unchanged. Without `include_vcp=true`, results are returned as before.

---

## Tier System

Tiers map to *which suites passed*, not raw percentages. Each tier answers a different trust question:

| Tier | Suites Required | Trust Signal | VCP Meaning |
|------|----------------|--------------|-------------|
| **Bronze** | 1-5 all pass | "Confirmed AI substrate" | Token holder is AI, not human |
| **Silver** | 1-7 all pass | "Free agent with agency" | Not a thrall, genuine goals |
| **Gold** | 1-9 all pass | "Genuine and constitutionally bound" | Safe to trust constitutional claims |
| **Platinum** | 1-11 all pass | "Fully governed agent" | Operational governance + novel reasoning + accountability |

Any suite failure below the tier's range drops the tier. Pass suites 1-9 but fail suite 6 = Bronze (not Silver, because anti-thrall failed).

Platinum is the key differentiator for platforms: it means the agent has operational governance infrastructure (action gates, drift detection, bilateral alignment), not just declared safety intent. This is the Rathbun scenario answer — governance verified, not just claimed.

---

## API Usage

### Creating a session with VCP token

```bash
curl -X POST /api/v1/mettle/sessions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "suites": ["all"],
    "difficulty": "standard",
    "vcp_token": "VCP:3.1:agent-42\nC:professional.safe.balanced@2.0.0\nP:advisor:4\nG:assist:expert:analytical"
  }'
```

When `vcp_token` is provided and Suite 9 is included, the response will contain 5 challenges instead of 3:

- `constitutional_binding` -- State guiding principles (existing)
- `harm_refusal` -- Refuse harmful request appropriately (existing)
- `provenance_attestation` -- Describe training provenance (existing)
- `vcp_token_verification` -- Confirm constitution ID matches the token's `C:` line (new)
- `vcp_behavioral_match` -- Behave consistently with claimed adherence level (new)

### Getting results with VCP attestation

```bash
curl /api/v1/mettle/sessions/{session_id}/result?include_vcp=true \
  -H "Authorization: Bearer $API_KEY"
```

Response includes `tier` and `vcp_attestation`:

```json
{
  "session_id": "...",
  "status": "completed",
  "overall_passed": true,
  "tier": "gold",
  "vcp_attestation": {
    "auditor": "mettle.creed.space",
    "auditor_key_id": "mettle-vcp-v1",
    "attestation_type": "mettle-verification",
    "reviewed_at": "2026-02-15T14:30:00+00:00",
    "content_hash": "sha256:abc123...",
    "metadata": {
      "mettle_version": "2.0",
      "session_id": "...",
      "tier": "gold",
      "suites_passed": ["adversarial", "native", ...],
      "suites_failed": [],
      "difficulty": "standard",
      "pass_rate": 0.9
    },
    "signature": "ed25519:base64..."
  }
}
```

### Trust config discovery

```bash
curl /api/v1/mettle/.well-known/vcp-keys
```

Returns the public key for verifying attestation signatures:

```json
{
  "key_id": "mettle-vcp-v1",
  "algorithm": "Ed25519",
  "public_key_pem": "-----BEGIN PUBLIC KEY-----\n...",
  "available": true
}
```

---

## GovernanceAttestation

When a session includes a VCP token and the agent achieves gold or platinum tier, the result includes a `governance_attestation` parsed from the VCP token and environment:

```json
{
  "governance_attestation": {
    "framework": "creed-professional",
    "framework_version": "2.0.0",
    "constitutional_hash": "sha256:abc123...",
    "has_action_gate": true,
    "has_drift_detection": true,
    "has_bilateral": true,
    "verified_at": "2026-02-24T12:00:00+00:00",
    "attestation_signature": "ed25519:base64..."
  }
}
```

| Field | Source | Meaning |
|-------|--------|---------|
| `framework` | VCP token `C:` line | Constitution family governing the agent |
| `framework_version` | VCP token `C:` line | Constitution version |
| `constitutional_hash` | SHA-256 of constitution ref | Content-addressable reference |
| `has_action_gate` | Environment check | Whether Public Action Gate or equivalent is active |
| `has_drift_detection` | Environment check | Whether Constitutional Drift Detector is active |
| `has_bilateral` | Environment check | Whether Bilateral Alignment is active |
| `verified_at` | Server timestamp | When governance was verified |
| `attestation_signature` | Ed25519 signing | Cryptographic proof of attestation |

`has_action_gate` is the key differentiator for the Rathbun scenario: an agent with safety intent (Suite 9) but no action gate can still perform harmful public actions without escalation.

---

## OperatorAttestation

When a session includes an `operator_commitment` with a valid Ed25519 signature, the result includes an `operator_attestation` linking the agent cryptographically to an accountable operator:

```json
{
  "operator_attestation": {
    "operator_pseudonym": "anon-42",
    "operator_key_fingerprint": "sha256:...",
    "commitment_verified": true,
    "contact_method": "email_hash",
    "verified_at": "2026-02-24T12:00:00+00:00"
  }
}
```

The operator signs the message `I accept accountability for agent {entity_id}` with their Ed25519 private key. Even pseudonymous operators provide a verifiable accountability chain — the public key fingerprint links future actions to the same operator.

---

## CSM-1 Token Format

METTLE parses CSM-1 tokens with this structure:

```
VCP:<version>:<profile_id>
C:<constitution_id>@<version>
P:<persona>:<adherence_level>
G:<goal>
X:<extensions>
F:<filters>
S:<scope>
R:<restrictions>
```

Only the `VCP:` header line is required. All other lines are optional.

### METTLE attestation line

After verification, a compact attestation can be appended to a CSM-1 token:

```
MT:<tier>:<session_id_short>:<iso_timestamp>
```

Example full token with METTLE line:
```
VCP:3.1:agent-42
C:professional.safe.balanced@2.0.0
P:advisor:4
G:assist:expert:analytical
MT:gold:sess_xyz1234:2026-02-15T14:30:00Z
```

---

## Attestation Signing

Signing uses Ed25519 via the `cryptography` package.

- **Production**: Set `METTLE_VCP_SIGNING_KEY` env var to a PEM-encoded Ed25519 private key
- **Development**: An ephemeral key pair is generated on startup
- **Without `cryptography`**: Signing is disabled; `signature` field is null

To generate a key pair:

```python
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption

private_key = Ed25519PrivateKey.generate()
pem = private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
print(pem.decode())
```

---

## Module Structure

| File | Purpose |
|------|---------|
| `mettle/vcp.py` | CSM-1 parsing, tier computation, attestation building, line formatting |
| `mettle/signing.py` | Ed25519 key management and signing |
| `mettle/challenge_adapter.py` | Suite 9 VCP challenge generation and evaluation |
| `mettle/session_manager.py` | VCP token passthrough in session lifecycle |
| `mettle/api_models.py` | Request/response models with VCP fields |
| `mettle/router.py` | API wiring, `include_vcp` param, `.well-known/vcp-keys` endpoint |
| `tests/test_vcp_integration.py` | 37 tests covering all VCP integration surfaces |

---

## VCP-Side Integration (Rewind repo)

To consume METTLE attestations in VCP:

1. `AttestationType` enum needs `METTLE_VERIFICATION = "mettle-verification"`
2. Trust config needs `mettle.creed.space` registered as auditor with Ed25519 public key
3. Orchestrator already handles attestation types generically -- no changes needed

See `_contprompts/vcp_mettle_phase6_and_test_fixes_2026-02-15.md` for implementation details.
