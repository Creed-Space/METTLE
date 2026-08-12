# Signing-key compromise

**Owner:** security incident commander. **Secondary:** credential operator and
release authority.

## Trigger

Private-key exposure, unauthorized signing, a public-key mismatch, unexplained
key-ID change, secret-manager access anomaly, leaked JWT secret, or credible
evidence that credentials were forged.

## Immediate safety

1. Set `METTLE_CREDENTIAL_ISSUANCE_ENABLED=false` through the provider secret and
   configuration authority. Confirm the resulting deployed SHA and configuration
   event. Do not rotate blindly while issuance continues.
2. Preserve key IDs, public fingerprints, access audit events, suspected first
   exposure, deployed candidates, and representative forged JTIs. Never copy the
   private key into the incident record.
3. Keep verification fail closed. Do not delete old public material until impact
   and overlap are understood.

## Scope

Determine whether the affected material is the Ed25519 issuer key, the JWT HS256
secret, a holder key, or more than one. Identify issuance times, credential
lifetimes, known JTIs, discovery caches, environments, and every service with
access. Treat a reused secret as compromised everywhere it appears.

## Recover

1. Generate replacement material in the authorized secret manager or hardware
   boundary. Use a new Ed25519 key ID.
2. Publish the new Ed25519 public key and an explicit compromised status for the
   old ID. Do not mark a compromised key merely verify-only without incident
   context.
3. Revoke affected online JTIs where they can be identified. For an HS256 secret,
   rotate it and regard all old quick badges as unverifiable.
4. Deploy once, verify discovery from an independent network, issue a synthetic
   credential, and reject tampering, expiry, old quick badges, and compromised
   JTIs.
5. Communicate the affected interval and offline-verification limitation. An
   Ed25519 signature already copied by a consumer cannot be recalled by database
   deletion.
6. Re-enable issuance only with security incident commander and release-authority
   approval.

## Close with evidence

Retain old and new public fingerprints, key IDs, compromise interval, secret
manager audit IDs, revocation counts, exact deployment SHA, independent consumer
results, communications, and retrospective actions.

Working if: no service signs with compromised material, discovery names the new
active key unambiguously, affected credentials fail through online policy, and
the public incident statement does not overclaim recall of offline signatures.
