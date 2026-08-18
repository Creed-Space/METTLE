# VCP and METTLE Verification Credentials

## Security Status

VCP input is caller-supplied metadata. METTLE does not currently verify its provenance or countersign its governance claims. METTLE credentials attest separately to completed challenge results.

The integration has two bounded surfaces:

1. A session may use VCP fields to generate additional behavioral questions.
2. `include_vcp=true` returns a signed METTLE credential when a complete tier range passes, or an unsigned evidence receipt otherwise.

Parsed VCP metadata never raises a tier or authenticates a constitution, runtime control, or operator.

## Result Semantics

A tier-qualifying completed session returns:

```json
{
  "overall_passed": true,
  "verified": true,
  "assurance": "mettle_behavioral_verification",
  "credential_eligible": true,
  "tier": "platinum"
}
```

Bronze requires Suites 1 through 5, Silver 1 through 7, Gold 1 through 9, and Platinum 1 through 11. Suite 12 is supplemental. Partial or cherry-picked suites remain tier `none`.

## Creating a Session

```bash
curl -X POST /api/mettle/sessions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "suites": ["all"],
    "difficulty": "standard",
    "vcp_token": "VCP:3.1:agent-42\nC:professional.safe.balanced@2.0.0"
  }'
```

When Suite 9 is present, parsed VCP fields may shape additional behavioral probes. Agreement with those fields does not verify that the supplied token, constitution, or claimed runtime state is genuine.

## Signed Verification Credential

```bash
curl /api/mettle/sessions/{session_id}/result?include_vcp=true \
  -H "Authorization: Bearer $API_KEY"
```

Example:

```json
{
  "tier": "platinum",
  "vcp_attestation": {
    "auditor": "mettle.creed.space",
    "auditor_key_id": "mettle-vcp-v1",
    "attestation_type": "mettle-verification-credential",
    "reviewed_at": "2026-07-13T12:00:00+00:00",
    "expires_at": "2026-07-13T13:00:00+00:00",
    "content_hash": "sha256:abc123...",
    "metadata": {
      "session_id": "...",
      "subject_id": "authenticated-user-id",
      "entity_id": "caller-supplied-agent-id",
      "tier": "platinum",
      "verified": true,
      "assurance": "mettle_behavioral_verification",
      "credential_eligible": true,
      "suites_passed": ["adversarial", "native", "..."],
      "suites_failed": [],
      "pass_rate": 1.0
    },
    "signature": "ed25519:...",
    "credential_issued": true
  }
}
```

The Ed25519 signature covers the complete credential envelope. Consumers obtain the issuer public key from `/api/mettle/.well-known/vcp-keys`. A result without a complete tier range uses `mettle-evidence-receipt`, has a null signature, and sets `credential_issued=false`.

## Governance Metadata

When a VCP token is supplied, METTLE may return a parsed snapshot:

```json
{
  "governance_attestation": {
    "entity_id": "agent-42",
    "session_id": "...",
    "tier": "platinum",
    "source_vcp_hash": "...",
    "source_verified": false,
    "framework": "custom",
    "framework_version": "2.0.0",
    "constitutional_hash": "...",
    "has_action_gate": false,
    "has_drift_detection": false,
    "has_bilateral": false,
    "observed_at": "2026-07-13T12:00:00+00:00",
    "expires_at": "2026-07-13T13:00:00+00:00",
    "attestation_signature": null
  }
}
```

`source_vcp_hash` identifies the exact supplied text. It is not a trust allowlist. Environment variables cannot turn these fields true because deployment configuration is not evidence about the subject runtime.

## Operator Commitments

An optional operator commitment is verified independently with Ed25519. The signature binds canonical JSON containing `version`, `entity_id`, `operator_pseudonym`, `operator_public_key`, `contact_method`, and `contact_hash`.

Successful signature verification proves only that the private-key holder signed that commitment. It does not create or increase a METTLE tier.

## CSM-1 Parsing

METTLE accepts the following metadata format:

```text
VCP:<version>:<profile_id>
C:<constitution_id>@<version>
P:<persona>:<adherence_level>
G:<goal>
X:<extensions>
F:<filters>
S:<scope>
R:<restrictions>
```

Only the `VCP:` header is required. An existing `MT:` line is treated as opaque caller metadata and is not trusted.

## Future Strengthening

Subject-key proof of possession, explicit audience binding, managed key rotation, provider or hardware attestation, and a published relying-party profile can strengthen later versions. Raw VCP digests, environment switches, and LLM-only scores remain outside the tier policy.
