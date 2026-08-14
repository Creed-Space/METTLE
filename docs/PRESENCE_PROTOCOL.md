# METTLE Presence Protocol v1

The Presence Protocol binds a METTLE session and its resulting credential to an
Ed25519 key controlled by the participant. It provides six concrete security
properties:

1. Every accepted submission is signed over the session, action, answers,
   current server nonce, and current transcript hash.
2. Every accepted submission rotates the nonce and advances a hash-chained
   transcript, blocking replay, answer substitution, and precomputed signed
   transcripts.
3. Suite challenges are disclosed sequentially. The next suite is released
   only after the current signed action is accepted, reducing harvesting and
   preventing whole-battery precomputation.
4. Every newly issued action carries a transcript-bound `mettle-continuity-v1`
   microchallenge. Its hidden family seed and the previous accepted signature
   make future interlocks unavailable for harvesting or precomputation.
5. A qualifying credential contains the holder public key, audience, unique
   credential identifier, final transcript commitment, and server-observed
   submission timing inside the issuer-signed envelope.
6. A verifier can demand a fresh holder signature before accepting the
   credential. The presentation challenge is verifier-owned, audience-bound,
   rate-limited, expires after 60 seconds, and can be used once.

Existing sessions remain compatible. Sessions without a `presence` registration
continue through the legacy bearer flow and do not gain holder-possession status.

## 1. Generate a holder key

The holder generates an Ed25519 key pair locally and keeps the private key. Send
only the SubjectPublicKeyInfo PEM public key to METTLE.

## 2. Create a key-bound session

```http
POST /api/mettle/sessions
Authorization: Bearer <api-key>
Content-Type: application/json

{
  "suites": ["adversarial", "native", "self-reference", "social", "inverse-turing"],
  "difficulty": "standard",
  "entity_id": "agent-42",
  "presence": {
    "public_key_pem": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----\n",
    "audience": "service.example"
  }
}
```

The response contains the first client-safe state:

```json
{
  "presence": {
    "protocol": "mettle-presence-v1",
    "key_fingerprint": "sha256:...",
    "audience": "service.example",
    "nonce": "...",
    "transcript_hash": "sha256:...",
    "sequence": 0,
    "action": "suite:adversarial",
    "completed": false,
    "continuity_protocol": "mettle-continuity-v1"
  }
}
```

The current state can also be recovered from `GET /api/mettle/sessions/{id}` by
the authenticated session owner.

For a Presence session, `challenges` contains only the first issued suite. Each
successful suite response returns `next_challenge` and updates
`presence.action`. Submitting an unissued suite or round fails before evaluation.
The issued suite also contains `_mettle_continuity`, with a challenge identifier,
32-bit starting value, and eight ordered operations. The reference
`scripts.testing.solver.solve_continuity_challenge` helper implements the public client
algorithm.

## 3. Sign every submission

Canonical JSON uses UTF-8, sorted object keys, no insignificant whitespace, and
literal Unicode. Hashes use lowercase SHA-256 with the `sha256:` prefix.

For a single-suite submission, hash the exact `answers` object and sign:

```json
{
  "action": "suite:adversarial",
  "nonce": "<current nonce>",
  "payload_hash": "sha256:<hash of canonical answers JSON>",
  "previous_transcript_hash": "sha256:<current transcript hash>",
  "protocol": "mettle-presence-v1",
  "purpose": "mettle-session-submission",
  "session_id": "<session id>"
}
```

For multi-round Suite 10 submissions, the action is `round:<number>`.

Submit the base64 Ed25519 signature with the answers:

```json
{
  "suite": "adversarial",
  "answers": {
    "q1": {"value": 42},
    "_mettle_continuity": {
      "challenge_id": "<current continuity challenge id>",
      "computed": 1234567890
    }
  },
  "presence_proof": {
    "nonce": "<current nonce>",
    "previous_transcript_hash": "sha256:<current transcript hash>",
    "signature": "<base64 Ed25519 signature>"
  }
}
```

The response contains the next nonce and transcript hash. A stale nonce, stale
transcript, wrong action, changed answer, wrong session, malformed signature, or
different private key causes rejection. A missing, stale, or incorrect continuity
answer also causes rejection before suite evaluation.

## 4. Obtain the credential

After a tier-qualifying session completes:

```http
GET /api/mettle/sessions/{id}/result?include_vcp=true
Authorization: Bearer <api-key>
```

The issuer-signed `mettle-presence-credential` contains:

* `metadata.jti`, the revocable credential identifier.
* `metadata.audience`, the intended verifier or service.
* `metadata.proof_of_possession.public_key_pem` and `key_fingerprint`.
* `metadata.proof_of_possession.transcript_hash` and `sequence`.
* `metadata.proof_of_possession.server_timing`, the signed sequence of actions
  and server-observed submission latencies.
* `metadata.proof_of_possession.continuity`, the signed family version, unique
  challenge count, transcript-binding statement, and observed maximum latency.

Server timing includes evaluation overhead and should be calibrated empirically
before applying relay-suspicion thresholds.

A Presence envelope is not a portable bearer credential. Generic Python, CLI,
JavaScript, and Rust credential verifiers reject
`mettle-presence-credential`, even when the issuer signature and status receipt
are valid. Acceptance requires the live holder path below. Callers that need a
portable bearer credential must use `mettle-verification-credential` instead.

## 5. Verify live holder possession

The verifier creates a one-use challenge:

```http
POST /api/mettle/presentation-challenges
Authorization: Bearer <verifier-api-key>
Content-Type: application/json

{
  "credential_jti": "<32 lowercase hex characters>",
  "audience": "service.example"
}
```

The holder signs canonical JSON with this shape:

```json
{
  "audience": "service.example",
  "challenge_id": "<challenge id>",
  "credential_jti": "<credential jti>",
  "expires_at": "<UTC timestamp normalized with Z>",
  "nonce": "<challenge nonce>",
  "protocol": "mettle-presence-v1",
  "purpose": "mettle-credential-presentation"
}
```

The verifier submits the credential and holder signature:

```http
POST /api/mettle/presentations/verify
Authorization: Bearer <same verifier-api-key>
Content-Type: application/json

{
  "challenge_id": "<challenge id>",
  "attestation": {"...": "complete issuer-signed credential"},
  "holder_signature": "<base64 Ed25519 signature>"
}
```

METTLE checks issuer integrity, credential expiry, tier integrity, audience,
revocation, challenge ownership, challenge expiry, one-use status, and the live
holder signature. Success consumes the challenge atomically.

## 6. Revoke a credential

An administrator can revoke a Presence credential through the shared durable
revocation namespace:

```http
POST /api/badge/revoke
X-Admin-Key: <admin-key>
Content-Type: application/json

{
  "jti": "<credential jti>",
  "entity_id": "agent-42",
  "reason": "Compromised holder key"
}
```

Future live presentations fail closed when the revocation store reports the JTI.

## Security boundary

Presence v1 establishes cryptographic continuity between session submissions,
the issued credential, and a later live presenter. It materially raises the cost
of credential copying, replay, answer substitution, transcript harvesting, and
simple relay. Continuity microchallenges force live, sequential computation and
prevent future interlocks from being collected before the preceding signed
response exists. Colluding parties can still share a private key or
operate a solver as a service. METTLE therefore records signed server timing and
both protocol versions so relay and solver-adaptation controls can be calibrated
and upgraded without silently changing the meaning of older credentials.

Working if: copying an issuer-signed Presence envelope and fresh status receipt
does not pass any portable verifier, while a fresh audience-bound challenge and
valid holder signature pass once and the same challenge then fails replay.
