#!/usr/bin/env node
/* Portable METTLE credential acceptance using issuer keys and signed status. */

const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const SCHEMA_VERSION = '1.1';
const POLICY_VERSION = '2026-08-14';
const CLOCK_SKEW_MS = 30_000;
const TIER_SUITES = [
  'adversarial', 'native', 'self-reference', 'social', 'inverse-turing',
  'anti-thrall', 'agency', 'counter-coaching', 'intent-provenance',
  'novel-reasoning', 'governance',
];
const ALL_SUITES = [...TIER_SUITES, 'llm-dynamic'];
const TIERS = { bronze: 5, silver: 7, gold: 9, platinum: 11 };
const STATUS_PROTOCOL = 'mettle-credential-status-v1';
const STATUS_TTL_MS = 300_000;
const STATUS_POINTER = {
  protocol: STATUS_PROTOCOL,
  endpoint: 'https://mettle.sh/api/mettle/credentials/status',
  method: 'POST',
};

function normalize(value) {
  if (Array.isArray(value)) return value.map(normalize);
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.keys(value).sort().map(key => [key, normalize(value[key])]),
    );
  }
  return value;
}

function canonicalBytes(value) {
  return Buffer.from(JSON.stringify(normalize(value)), 'utf8');
}

function expectedTier(passed) {
  const set = new Set(passed);
  let result = 'none';
  for (const [tier, count] of Object.entries(TIERS)) {
    if (TIER_SUITES.slice(0, count).every(suite => set.has(suite))) result = tier;
  }
  return result;
}

function validSuiteList(value) {
  return Array.isArray(value)
    && value.every(item => typeof item === 'string' && ALL_SUITES.includes(item))
    && new Set(value).size === value.length;
}

function validPresence(metadata) {
  const proof = metadata.proof_of_possession;
  if (!proof || typeof proof !== 'object' || Array.isArray(proof)) return false;
  if (typeof metadata.audience !== 'string' || metadata.audience.length === 0
      || proof.protocol !== 'mettle-presence-v1'
      || typeof proof.public_key_pem !== 'string'
      || !/^sha256:[0-9a-f]{64}$/.test(proof.key_fingerprint || '')
      || !/^sha256:[0-9a-f]{64}$/.test(proof.transcript_hash || '')
      || !Number.isInteger(proof.sequence) || proof.sequence <= 0) return false;
  let fingerprint;
  try {
    const der = crypto.createPublicKey(proof.public_key_pem)
      .export({ type: 'spki', format: 'der' });
    fingerprint = `sha256:${crypto.createHash('sha256').update(der).digest('hex')}`;
  } catch {
    return false;
  }
  if (fingerprint !== proof.key_fingerprint) return false;
  const timing = proof.server_timing;
  if (!timing || !Number.isInteger(timing.total_elapsed_ms)
      || timing.total_elapsed_ms < 0 || !Array.isArray(timing.submissions)
      || timing.submissions.length !== proof.sequence) return false;
  const continuity = proof.continuity;
  const challengeIds = new Set();
  for (let index = 0; index < timing.submissions.length; index += 1) {
    const item = timing.submissions[index];
    if (!item || item.sequence !== index + 1
        || typeof item.action !== 'string'
        || !/^(suite|round):/.test(item.action)
        || !Number.isInteger(item.response_time_ms) || item.response_time_ms < 0
        || !/^sha256:[0-9a-f]{64}$/.test(item.transcript_hash || '')) return false;
    if (continuity) {
      if (item.challenge_family !== 'mettle-continuity-v1'
          || !/^[0-9a-f]{32}$/.test(item.challenge_id || '')) return false;
      challengeIds.add(item.challenge_id);
    }
  }
  if (timing.submissions.at(-1)?.transcript_hash !== proof.transcript_hash) return false;
  return !continuity || (
    continuity.protocol === 'mettle-continuity-v1'
    && continuity.challenge_count === proof.sequence
    && continuity.transcript_bound === true
    && Number.isInteger(continuity.max_response_time_ms)
    && continuity.max_response_time_ms >= 0
    && challengeIds.size === proof.sequence
  );
}

function verifyStatus(statusReceipt, keyring, credentialJti, verificationTime) {
  if (!statusReceipt || typeof statusReceipt !== 'object'
      || Array.isArray(statusReceipt)) return false;
  const expectedFields = [
    'auditor', 'auditor_key_id', 'credential_jti', 'expires_at',
    'observed_at', 'protocol', 'signature', 'status',
  ];
  if (JSON.stringify(Object.keys(statusReceipt).sort())
      !== JSON.stringify(expectedFields)) return false;
  if (statusReceipt.protocol !== STATUS_PROTOCOL
      || statusReceipt.auditor !== 'mettle.creed.space'
      || statusReceipt.credential_jti !== credentialJti
      || statusReceipt.status !== 'good') return false;
  const observedAt = Date.parse(statusReceipt.observed_at);
  const expiresAt = Date.parse(statusReceipt.expires_at);
  const now = Date.parse(verificationTime);
  if (![observedAt, expiresAt, now].every(Number.isFinite)
      || expiresAt - observedAt !== STATUS_TTL_MS
      || expiresAt <= observedAt || observedAt > now + CLOCK_SKEW_MS
      || expiresAt + CLOCK_SKEW_MS <= now) return false;
  const publicKey = keyring[statusReceipt.auditor_key_id];
  const encodedSignature = statusReceipt.signature;
  if (!publicKey || typeof encodedSignature !== 'string'
      || !encodedSignature.startsWith('ed25519:')) return false;
  const unsigned = { ...statusReceipt };
  delete unsigned.signature;
  try {
    return crypto.verify(
      null,
      canonicalBytes(unsigned),
      crypto.createPublicKey(publicKey),
      Buffer.from(encodedSignature.slice('ed25519:'.length), 'base64'),
    );
  } catch {
    return false;
  }
}

function verifyCredential(attestation, statusReceipt, keyring, verificationTime) {
  const metadata = attestation.metadata;
  if (!metadata || attestation.credential_issued !== true) return false;
  if (attestation.attestation_type !== 'mettle-verification-credential') return false;
  if (metadata.credential_schema_version !== SCHEMA_VERSION
      || metadata.suite_policy_version !== POLICY_VERSION) return false;
  const passed = metadata.suites_passed;
  const failed = metadata.suites_failed;
  const supplemental = metadata.supplemental_suites_passed || [];
  if (![passed, failed, supplemental].every(validSuiteList)) return false;
  if (passed.some(item => failed.includes(item))
      || supplemental.some(item => passed.includes(item) || failed.includes(item))) return false;
  if (expectedTier(passed) !== metadata.tier
      || metadata.credential_eligible !== true
      || typeof metadata.session_id !== 'string' || metadata.session_id.length === 0
      || typeof metadata.subject_id !== 'string' || metadata.subject_id.length === 0
      || !/^[0-9a-f]{32}$/.test(metadata.jti || '')
      || !metadata.credential_status
      || metadata.credential_status.protocol !== STATUS_POINTER.protocol
      || metadata.credential_status.endpoint !== STATUS_POINTER.endpoint
      || metadata.credential_status.method !== STATUS_POINTER.method
      || Object.keys(metadata.credential_status).length !== 3
      || (metadata.entity_id != null
        && metadata.entity_id_binding !== 'self_asserted_by_authenticated_subject')) return false;
  if (attestation.attestation_type === 'mettle-presence-credential'
      && !validPresence(metadata)) return false;
  const expectedHash = `sha256:${crypto.createHash('sha256')
    .update(canonicalBytes(metadata)).digest('hex')}`;
  if (attestation.content_hash !== expectedHash) return false;
  const reviewedAt = Date.parse(attestation.reviewed_at);
  const expiresAt = Date.parse(attestation.expires_at);
  const now = Date.parse(verificationTime);
  if (![reviewedAt, expiresAt, now].every(Number.isFinite)) return false;
  if (expiresAt <= reviewedAt || reviewedAt > now + CLOCK_SKEW_MS
      || expiresAt + CLOCK_SKEW_MS <= now) return false;
  const publicKey = keyring[attestation.auditor_key_id];
  const encodedSignature = attestation.signature;
  if (!publicKey || typeof encodedSignature !== 'string'
      || !encodedSignature.startsWith('ed25519:')) return false;
  const unsigned = { ...attestation };
  delete unsigned.signature;
  let signatureValid;
  try {
    signatureValid = crypto.verify(
      null,
      canonicalBytes(unsigned),
      crypto.createPublicKey(publicKey),
      Buffer.from(encodedSignature.slice('ed25519:'.length), 'base64'),
    );
  } catch {
    return false;
  }
  return signatureValid
    && verifyStatus(statusReceipt, keyring, metadata.jti, verificationTime);
}

function main() {
  const fixturePath = process.argv[2]
    || path.join(__dirname, '..', 'fixtures', 'credentials', 'v1.json');
  const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
  const keyring = { [fixture.key.key_id]: fixture.key.public_key_pem };
  for (const testCase of fixture.cases) {
    const actual = verifyCredential(
      testCase.attestation, testCase.status_receipt, keyring,
      testCase.verification_time,
    );
    if (actual !== testCase.expected_valid) {
      throw new Error(`${testCase.name}: expected ${testCase.expected_valid}, got ${actual}`);
    }
  }
  console.log(`Verified ${fixture.cases.length} JavaScript compatibility cases`);
}

if (require.main === module) main();

module.exports = { canonicalBytes, verifyCredential, verifyStatus };
