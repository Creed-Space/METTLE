//! Offline METTLE fixture verifier for Rust consumers.
//!
//! Cargo dependencies:
//! base64 = "0.22"
//! chrono = "0.4"
//! ed25519-dalek = { version = "2", features = ["pkcs8", "pem"] }
//! serde_json = "1"
//! sha2 = "0.10"

use base64::{engine::general_purpose::STANDARD, Engine};
use chrono::{DateTime, Duration, Utc};
use ed25519_dalek::pkcs8::DecodePublicKey;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, env, fs};

const CLOCK_SKEW_SECONDS: i64 = 30;

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

fn verify(attestation: &Value, public_key_pem: &str, at: DateTime<Utc>) -> bool {
    let metadata = match attestation.get("metadata") {
        Some(Value::Object(_)) => &attestation["metadata"],
        _ => return false,
    };
    if metadata["credential_schema_version"] != "1.0"
        || metadata["suite_policy_version"] != "2026-08-12"
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
    let skew = Duration::seconds(CLOCK_SKEW_SECONDS);
    if expires <= reviewed || reviewed > at + skew || expires + skew <= at {
        return false;
    }
    let encoded = match attestation["signature"]
        .as_str()
        .and_then(|value| value.strip_prefix("ed25519:"))
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
    let mut unsigned = attestation.clone();
    unsigned
        .as_object_mut()
        .expect("attestation object")
        .remove("signature");
    key.verify(&canonical_bytes(&unsigned), &signature).is_ok()
}

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "fixtures/credentials/v1.json".into());
    let fixture: Value =
        serde_json::from_str(&fs::read_to_string(path).expect("fixture")).expect("fixture JSON");
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
        let actual = verify(&case["attestation"], public_key, at);
        assert_eq!(
            actual,
            case["expected_valid"].as_bool().expect("expected"),
            "fixture case {}",
            case["name"].as_str().unwrap_or("unnamed")
        );
    }
    println!("Verified {} Rust compatibility cases", cases.len());
}
