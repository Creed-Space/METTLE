//! Portable METTLE credential acceptance for Rust consumers.
//!
//! The verifier requires a pinned issuer key and a fresh issuer-signed status
//! receipt. Run it through the pinned manifest in `examples/Cargo.toml`.

use base64::{engine::general_purpose::STANDARD, Engine};
use chrono::{DateTime, Duration, Utc};
use ed25519_dalek::pkcs8::{DecodePublicKey, EncodePublicKey};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::{collections::{BTreeMap, BTreeSet}, env, fs};

const CLOCK_SKEW_SECONDS: i64 = 30;
const STATUS_TTL_SECONDS: i64 = 300;
const STATUS_PROTOCOL: &str = "mettle-credential-status-v1";
const STATUS_ENDPOINT: &str = "https://mettle.sh/api/mettle/credentials/status";
const TIER_SUITES: [&str; 11] = [
    "adversarial",
    "native",
    "self-reference",
    "social",
    "inverse-turing",
    "anti-thrall",
    "agency",
    "counter-coaching",
    "intent-provenance",
    "novel-reasoning",
    "governance",
];
const ALL_SUITES: [&str; 12] = [
    "adversarial",
    "native",
    "self-reference",
    "social",
    "inverse-turing",
    "anti-thrall",
    "agency",
    "counter-coaching",
    "intent-provenance",
    "novel-reasoning",
    "governance",
    "llm-dynamic",
];

fn normalize(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(normalize).collect()),
        Value::Object(values) => {
            let sorted: BTreeMap<_, _> = values
                .iter()
                .map(|(key, value)| (key.clone(), normalize(value)))
                .collect();
            Value::Object(sorted.into_iter().collect::<Map<String, Value>>())
        }
        Value::Number(number) => match number.as_f64() {
            Some(value)
                if number.as_i64().is_none()
                    && number.as_u64().is_none()
                    && value.is_finite()
                    && value.fract() == 0.0
                    && value >= i64::MIN as f64
                    && value <= i64::MAX as f64 =>
            {
                Value::Number(Number::from(value as i64))
            }
            _ => value.clone(),
        },
        _ => value.clone(),
    }
}

fn canonical_bytes(value: &Value) -> Vec<u8> {
    serde_json::to_vec(&normalize(value)).expect("canonical JSON")
}

fn lowercase_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_id(value: &Value) -> bool {
    value
        .as_str()
        .is_some_and(|text| text.starts_with("sha256:") && lowercase_hex(&text[7..], 64))
}

fn suite_set(value: &Value) -> Option<BTreeSet<String>> {
    let array = value.as_array()?;
    let mut result = BTreeSet::new();
    for item in array {
        let suite = item.as_str()?;
        if !ALL_SUITES.contains(&suite) || !result.insert(suite.to_owned()) {
            return None;
        }
    }
    Some(result)
}

fn expected_tier(passed: &BTreeSet<String>) -> &'static str {
    let mut result = "none";
    for (tier, count) in [("bronze", 5), ("silver", 7), ("gold", 9), ("platinum", 11)] {
        if TIER_SUITES[..count]
            .iter()
            .all(|suite| passed.contains(*suite))
        {
            result = tier;
        }
    }
    result
}

fn time_window_valid(reviewed: DateTime<Utc>, expires: DateTime<Utc>, at: DateTime<Utc>) -> bool {
    let skew = Duration::seconds(CLOCK_SKEW_SECONDS);
    expires > reviewed && reviewed <= at + skew && expires + skew > at
}

fn signature_valid(value: &Value, expected_key_id: &str, public_key_pem: &str) -> bool {
    if value["auditor_key_id"].as_str() != Some(expected_key_id) {
        return false;
    }
    let encoded = match value["signature"]
        .as_str()
        .and_then(|signature| signature.strip_prefix("ed25519:"))
    {
        Some(value) => value,
        None => return false,
    };
    let signature = match STANDARD
        .decode(encoded)
        .ok()
        .and_then(|bytes| Signature::from_slice(&bytes).ok())
    {
        Some(value) => value,
        None => return false,
    };
    let key = match VerifyingKey::from_public_key_pem(public_key_pem) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let mut unsigned = value.clone();
    match unsigned.as_object_mut() {
        Some(object) => {
            object.remove("signature");
        }
        None => return false,
    }
    key.verify(&canonical_bytes(&unsigned), &signature).is_ok()
}

fn presence_valid(metadata: &Value) -> bool {
    let proof = match metadata.get("proof_of_possession") {
        Some(Value::Object(_)) => &metadata["proof_of_possession"],
        _ => return false,
    };
    if metadata["audience"].as_str().is_none_or(str::is_empty)
        || proof["protocol"] != "mettle-presence-v1"
        || !sha256_id(&proof["key_fingerprint"])
        || !sha256_id(&proof["transcript_hash"])
    {
        return false;
    }
    let sequence = match proof["sequence"].as_u64() {
        Some(value) if value > 0 => value,
        _ => return false,
    };
    let public_key_pem = match proof["public_key_pem"].as_str() {
        Some(value) => value,
        None => return false,
    };
    let key = match VerifyingKey::from_public_key_pem(public_key_pem) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let der = match key.to_public_key_der() {
        Ok(value) => value,
        Err(_) => return false,
    };
    if format!("sha256:{:x}", Sha256::digest(der.as_bytes()))
        != proof["key_fingerprint"]
    {
        return false;
    }
    let timing = &proof["server_timing"];
    let submissions = match timing["submissions"].as_array() {
        Some(value)
            if timing["total_elapsed_ms"].as_i64().is_some_and(|n| n >= 0)
                && value.len() == sequence as usize => value,
        _ => return false,
    };
    let continuity = proof.get("continuity");
    let mut challenge_ids = BTreeSet::new();
    for (index, submission) in submissions.iter().enumerate() {
        let action = submission["action"].as_str().unwrap_or("");
        if submission["sequence"].as_u64() != Some(index as u64 + 1)
            || !(action.starts_with("suite:") || action.starts_with("round:"))
            || !submission["response_time_ms"].as_i64().is_some_and(|n| n >= 0)
            || !sha256_id(&submission["transcript_hash"])
        {
            return false;
        }
        if continuity.is_some() {
            let challenge_id = submission["challenge_id"].as_str().unwrap_or("");
            if submission["challenge_family"] != "mettle-continuity-v1"
                || !lowercase_hex(challenge_id, 32)
            {
                return false;
            }
            challenge_ids.insert(challenge_id.to_owned());
        }
    }
    if submissions
        .last()
        .and_then(|item| item["transcript_hash"].as_str())
        != proof["transcript_hash"].as_str()
    {
        return false;
    }
    match continuity {
        None => true,
        Some(value) => {
            value["protocol"] == "mettle-continuity-v1"
                && value["challenge_count"].as_u64() == Some(sequence)
                && value["transcript_bound"].as_bool() == Some(true)
                && value["max_response_time_ms"].as_i64().is_some_and(|n| n >= 0)
                && challenge_ids.len() == sequence as usize
        }
    }
}

fn status_valid(
    receipt: &Value,
    expected_key_id: &str,
    public_key_pem: &str,
    credential_jti: &str,
    at: DateTime<Utc>,
) -> bool {
    let object = match receipt.as_object() {
        Some(value) => value,
        None => return false,
    };
    let expected_fields: BTreeSet<&str> = [
        "auditor", "auditor_key_id", "credential_jti", "expires_at",
        "observed_at", "protocol", "signature", "status",
    ]
    .into_iter()
    .collect();
    if object.keys().map(String::as_str).collect::<BTreeSet<_>>() != expected_fields
        || receipt["protocol"] != STATUS_PROTOCOL
        || receipt["auditor"] != "mettle.creed.space"
        || receipt["credential_jti"] != credential_jti
        || receipt["status"] != "good"
    {
        return false;
    }
    let observed = match receipt["observed_at"]
        .as_str()
        .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
    {
        Some(value) => value.with_timezone(&Utc),
        None => return false,
    };
    let expires = match receipt["expires_at"]
        .as_str()
        .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
    {
        Some(value) => value.with_timezone(&Utc),
        None => return false,
    };
    expires - observed == Duration::seconds(STATUS_TTL_SECONDS)
        && time_window_valid(observed, expires, at)
        && signature_valid(receipt, expected_key_id, public_key_pem)
}

fn verify(
    attestation: &Value,
    status_receipt: &Value,
    expected_key_id: &str,
    public_key_pem: &str,
    at: DateTime<Utc>,
) -> bool {
    let metadata = match attestation.get("metadata") {
        Some(Value::Object(_)) => &attestation["metadata"],
        _ => return false,
    };
    if attestation["credential_issued"].as_bool() != Some(true)
        || attestation["attestation_type"] != "mettle-verification-credential"
        || metadata["credential_schema_version"] != "1.1"
        || metadata["suite_policy_version"] != "2026-08-14"
        || metadata["credential_eligible"].as_bool() != Some(true)
        || metadata["session_id"].as_str().is_none_or(str::is_empty)
        || metadata["subject_id"].as_str().is_none_or(str::is_empty)
    {
        return false;
    }
    let passed = match suite_set(&metadata["suites_passed"]) {
        Some(value) => value,
        None => return false,
    };
    let failed = match suite_set(&metadata["suites_failed"]) {
        Some(value) => value,
        None => return false,
    };
    let supplemental = match metadata.get("supplemental_suites_passed") {
        Some(value) => match suite_set(value) {
            Some(value) => value,
            None => return false,
        },
        None => BTreeSet::new(),
    };
    if !passed.is_disjoint(&failed)
        || !supplemental.is_disjoint(&passed)
        || !supplemental.is_disjoint(&failed)
        || metadata["tier"].as_str() != Some(expected_tier(&passed))
    {
        return false;
    }
    let credential_jti = metadata["jti"].as_str().unwrap_or("");
    let status_pointer = &metadata["credential_status"];
    if !lowercase_hex(credential_jti, 32)
        || status_pointer["protocol"] != STATUS_PROTOCOL
        || status_pointer["endpoint"] != STATUS_ENDPOINT
        || status_pointer["method"] != "POST"
        || status_pointer.as_object().is_none_or(|value| value.len() != 3)
        || (!metadata["entity_id"].is_null()
            && metadata["entity_id_binding"]
                != "self_asserted_by_authenticated_subject")
        || (attestation["attestation_type"] == "mettle-presence-credential"
            && !presence_valid(metadata))
    {
        return false;
    }
    let expected_hash = format!("sha256:{:x}", Sha256::digest(canonical_bytes(metadata)));
    if attestation["content_hash"] != expected_hash {
        return false;
    }
    let reviewed = match attestation["reviewed_at"]
        .as_str()
        .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
    {
        Some(value) => value.with_timezone(&Utc),
        None => return false,
    };
    let expires = match attestation["expires_at"]
        .as_str()
        .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
    {
        Some(value) => value.with_timezone(&Utc),
        None => return false,
    };
    time_window_valid(reviewed, expires, at)
        && signature_valid(attestation, expected_key_id, public_key_pem)
        && status_valid(
            status_receipt,
            expected_key_id,
            public_key_pem,
            credential_jti,
            at,
        )
}

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "fixtures/credentials/v1.json".into());
    let fixture: Value =
        serde_json::from_str(&fs::read_to_string(path).expect("fixture")).expect("fixture JSON");
    let key_id = fixture["key"]["key_id"].as_str().expect("key ID");
    let public_key = fixture["key"]["public_key_pem"]
        .as_str()
        .expect("public key");
    let cases = fixture["cases"].as_array().expect("cases");
    for case in cases {
        let at = DateTime::parse_from_rfc3339(
            case["verification_time"]
                .as_str()
                .expect("verification time"),
        )
        .expect("timestamp")
        .with_timezone(&Utc);
        let actual = verify(
            &case["attestation"],
            &case["status_receipt"],
            key_id,
            public_key,
            at,
        );
        assert_eq!(
            actual,
            case["expected_valid"].as_bool().expect("expected"),
            "fixture case {}",
            case["name"].as_str().unwrap_or("unnamed")
        );
    }
    println!("Verified {} Rust compatibility cases", cases.len());
}
