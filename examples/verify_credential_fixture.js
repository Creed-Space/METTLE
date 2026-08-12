#!/usr/bin/env node
/* Offline METTLE credential verifier using only Node.js standard modules. */

const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const SCHEMA_VERSION = '1.0';
const POLICY_VERSION = '2026-08-12';
const CLOCK_SKEW_MS = 30_000;
const SUITES = [
  'adversarial', 'native', 'self-reference', 'social', 'inverse-turing',
  'anti-thrall', 'agency', 'counter-coaching', 'intent-provenance',
  'novel-reasoning', 'governance',
];
const TIERS = { bronze: 5, silver: 7, gold: 9, platinum: 11 };

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
    if (SUITES.slice(0, count).every(suite => set.has(suite))) result = tier;
  }
  return result;
}

function verifyCredential(attestation, keyring, verificationTime) {
  const metadata = attestation.metadata;
  if (!metadata || attestation.credential_issued !== true) return false;
  if (!['mettle-verification-credential', 'mettle-presence-credential']
    .includes(attestation.attestation_type)) return false;
  if (metadata.credential_schema_version !== SCHEMA_VERSION
      || metadata.suite_policy_version !== POLICY_VERSION) return false;
  if (expectedTier(metadata.suites_passed || []) !== metadata.tier) return false;
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
  return crypto.verify(
    null,
    canonicalBytes(unsigned),
    crypto.createPublicKey(publicKey),
    Buffer.from(encodedSignature.slice('ed25519:'.length), 'base64'),
  );
}

function main() {
  const fixturePath = process.argv[2]
    || path.join(__dirname, '..', 'fixtures', 'credentials', 'v1.json');
  const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
  const keyring = { [fixture.key.key_id]: fixture.key.public_key_pem };
  for (const testCase of fixture.cases) {
    const actual = verifyCredential(
      testCase.attestation, keyring, testCase.verification_time,
    );
    if (actual !== testCase.expected_valid) {
      throw new Error(`${testCase.name}: expected ${testCase.expected_valid}, got ${actual}`);
    }
  }
  console.log(`Verified ${fixture.cases.length} JavaScript compatibility cases`);
}

if (require.main === module) main();

module.exports = { canonicalBytes, verifyCredential };
