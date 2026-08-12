# Public-key publication and rotation

**Owner:** credential operator. **Secondary:** release authority and independent
consumer reviewer.

## Trigger

Initial production activation, scheduled Ed25519 rotation, key-ID correction, or
recovery from a signing incident.

## Prepare

1. Generate the private key inside the authorized secret-management boundary.
2. Compute the public-key SHA-256 fingerprint independently in two tools or
   implementations. Retain only public material and the fingerprint in evidence.
3. Choose a new bounded key ID. Never place changed private material under an
   existing published ID.
4. Add the prior public key to `METTLE_VCP_VERIFYING_KEYS` for overlap.
5. Bind the release candidate, credential schema, suite policy, and key-change
   statement in the release manifest.

## Publish and validate

1. Deploy the private key and key ID together through provider secrets.
2. Fetch `GET /api/mettle/.well-known/vcp-keys` from an independent network.
3. Confirm the active ID and fingerprint, Ed25519 algorithm, schema and policy,
   and verify-only overlap entry.
4. Issue a synthetic credential on the deployed SHA. Verify it in Python,
   JavaScript, and Rust without server-private state.
5. Verify an unexpired old credential during overlap, reject a modified envelope,
   reject exact expiry, and test online revocation.
6. Publish the key-history row and receipt. Keep the old key through the longest
   old expiry plus clock allowance and cache margin.

## Emergency differences

A compromised key is labelled compromised, not merely retired. Stop issuance
first and follow `SIGNING_KEY_COMPROMISE.md`. Do not preserve trust solely to make
old verification green.

## Close with evidence

Retain public PEMs, fingerprints, key IDs, secret-manager event IDs without
secret values, deployed SHA, discovery response digest, fixture and live consumer
results, overlap start and end, and publication URL.

Working if: every verifier selects by signed key ID, two independent fingerprint
calculations agree, old and new safe credentials verify only within policy, and
the repository key-history table links to a production receipt.
