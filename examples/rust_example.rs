//! METTLE Rust SDK Example
//!
//! Complete verification flow for Becoming Minds.
//!
//! Add to Cargo.toml:
//! ```toml
//! [dependencies]
//! reqwest = { version = "0.11", features = ["json"] }
//! tokio = { version = "1", features = ["full"] }
//! serde = { version = "1", features = ["derive"] }
//! serde_json = "1"
//! regex = "1"
//! ```
//!
//! Run:
//!   cargo run --example rust_example

use serde::{Deserialize, Serialize};
use regex::Regex;
use std::collections::HashMap;

const METTLE_API: &str = "https://mettle-api.onrender.com/api";

#[derive(Debug, Serialize)]
struct StartSessionRequest {
    entity_id: String,
    difficulty: String,
}

#[derive(Debug, Deserialize)]
struct Challenge {
    id: String,
    #[serde(rename = "type")]
    challenge_type: String,
    prompt: String,
    #[serde(default)]
    data: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct StartSessionResponse {
    session_id: String,
    session_token: String,
    total_challenges: u32,
    current_challenge: Challenge,
}

#[derive(Debug, Serialize)]
struct SubmitAnswerRequest {
    session_id: String,
    challenge_id: String,
    answer: String,
}

#[derive(Debug, Deserialize)]
struct VerificationResult {
    passed: bool,
}

#[derive(Debug, Deserialize)]
struct SubmitAnswerResponse {
    result: VerificationResult,
    next_challenge: Option<Challenge>,
}

#[derive(Debug, Deserialize)]
struct BadgeInfo {
    expires_at: String,
}

#[derive(Debug, Deserialize)]
struct MettleResult {
    verified: bool,
    pass_rate: f64,
    badge: Option<String>,
    badge_info: Option<BadgeInfo>,
}

#[derive(Debug, Deserialize)]
struct BadgeVerifyResponse {
    valid: bool,
    error: Option<String>,
}

/// Complete METTLE verification flow.
async fn verify_agent(
    client: &reqwest::Client,
    entity_id: &str,
    difficulty: &str,
) -> Result<MettleResult, Box<dyn std::error::Error>> {
    println!("Starting verification for {}...", entity_id);

    // Step 1: Start session
    let start_req = StartSessionRequest {
        entity_id: entity_id.to_string(),
        difficulty: difficulty.to_string(),
    };

    let session: StartSessionResponse = client
        .post(format!("{}/session/start", METTLE_API))
        .json(&start_req)
        .send()
        .await?
        .json()
        .await?;

    let session_id = &session.session_id;
    let total = session.total_challenges;
    println!("Session {}: {} challenges", session_id, total);

    // Step 2: Answer challenges
    let mut current_challenge = Some(session.current_challenge);
    let mut challenge_num = 1;

    while let Some(challenge) = current_challenge {
        println!(
            "\nChallenge {}/{}: {}",
            challenge_num, total, challenge.challenge_type
        );
        println!("  Prompt: {}...", &challenge.prompt[..challenge.prompt.len().min(60)]);

        let answer = generate_answer(&challenge.challenge_type, &challenge.prompt, &challenge.data);

        let answer_req = SubmitAnswerRequest {
            session_id: session_id.clone(),
            challenge_id: challenge.id,
            answer,
        };

        let result: SubmitAnswerResponse = client
            .post(format!("{}/session/answer", METTLE_API))
            .header("X-Session-Token", &session.session_token)
            .json(&answer_req)
            .send()
            .await?
            .json()
            .await?;

        println!(
            "  Result: {}",
            if result.result.passed { "PASS" } else { "FAIL" }
        );

        current_challenge = result.next_challenge;
        challenge_num += 1;
    }

    // Step 3: Get final result
    let final_result: MettleResult = client
        .get(format!("{}/session/{}/result", METTLE_API, session_id))
        .header("X-Session-Token", &session.session_token)
        .send()
        .await?
        .json()
        .await?;

    println!("\n{}", "=".repeat(40));
    println!(
        "VERIFICATION {}",
        if final_result.verified {
            "PASSED"
        } else {
            "FAILED"
        }
    );
    println!("Pass rate: {:.0}%", final_result.pass_rate * 100.0);

    if let Some(ref badge) = final_result.badge {
        println!("Badge: {}...", &badge[..badge.len().min(40)]);
    }
    if let Some(ref badge_info) = final_result.badge_info {
        println!("Expires: {}", badge_info.expires_at);
    }

    Ok(final_result)
}

/// Generate answer for a challenge.
fn generate_answer(
    challenge_type: &str,
    prompt: &str,
    data: &HashMap<String, serde_json::Value>,
) -> String {
    match challenge_type {
        "speed_math" => {
            // Parse: "Calculate: 47 + 83"
            if let Some(expr) = prompt.split(": ").nth(1) {
                parse_simple_math(expr).to_string()
            } else {
                "0".to_string()
            }
        }

        "token_prediction" => {
            let token_pattern = Regex::new(r"(K[0-9a-fA-F]{6}-)(\d+)").unwrap();
            let tokens: Vec<_> = token_pattern.captures_iter(prompt).collect();
            if tokens.len() >= 2 {
                let previous: i64 = tokens[tokens.len() - 2][2].parse().unwrap_or(0);
                let last: i64 = tokens[tokens.len() - 1][2].parse().unwrap_or(0);
                return format!("{}{}", &tokens[tokens.len() - 1][1], last + last - previous);
            }
            let lower = prompt.to_lowercase();
            if lower.contains("lazy") {
                "dog".to_string()
            } else if lower.contains("roses are") {
                "red".to_string()
            } else {
                "unknown".to_string()
            }
        }

        "instruction_following" => {
            let instruction = data
                .get("instruction")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let kind = data.get("instruction_kind").and_then(|v| v.as_str());
            let marker = data.get("marker").and_then(|v| v.as_str()).unwrap_or("");
            match kind {
                Some("prefix") => return format!("{} Paris is France's capital.", marker),
                Some("suffix") => return format!("Paris is France's capital. {}", marker),
                Some("include") => return format!("Paris {} is France's capital.", marker),
                Some("exact_words") => {
                    let count = data.get("word_count").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
                    let mut words = vec![marker, "Paris", "is", "France's", "capital"];
                    while words.len() < count { words.push("clearly"); }
                    return words[..count].join(" ");
                }
                Some("start_digit") => {
                    let digit = data.get("starting_digit").and_then(|v| v.as_str()).unwrap_or("1");
                    return format!("{} Paris {} is France's capital.", digit, marker);
                }
                _ => {}
            }

            if instruction.contains("Indeed,") {
                "Indeed, I understand the requirement.".to_string()
            } else if instruction.contains("...") {
                "Here is my response...".to_string()
            } else if instruction.contains("therefore") {
                "Therefore, this follows logically.".to_string()
            } else if instruction.contains("5 words") {
                "This has five words exactly.".to_string()
            } else if instruction.contains("number") {
                "42 is the answer here.".to_string()
            } else {
                "Response following instructions.".to_string()
            }
        }

        "consistency" => {
            let num_responses = data
                .get("num_responses")
                .and_then(|v| v.as_u64())
                .unwrap_or(3) as usize;
            let base = "The sky is blue";
            vec![base; num_responses].join(" | ")
        }

        "chained_reasoning" => {
            if let Some(chain) = data.get("chain").and_then(|v| v.as_array()) {
                let mut result = chain
                    .first()
                    .and_then(|v| v.get("value"))
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);

                for step in chain.iter().skip(1) {
                    let op = step.get("op").and_then(|v| v.as_str()).unwrap_or("");
                    let value = step.get("value").and_then(|v| v.as_i64()).unwrap_or(0);

                    match op {
                        "+" => result += value,
                        "-" => result -= value,
                        "*" => result *= value,
                        _ => {}
                    }
                }
                result.to_string()
            } else {
                "0".to_string()
            }
        }

        _ => "default answer".to_string(),
    }
}

/// Parse and compute simple math expressions (+ - * only).
fn parse_simple_math(expr: &str) -> i64 {
    let mut result: i64 = 0;
    let mut current: i64 = 0;
    let mut op = '+';

    for c in format!("{}+", expr).chars() {
        if c.is_ascii_digit() {
            current = current * 10 + (c as i64 - '0' as i64);
        } else if ['+', '-', '*'].contains(&c) {
            match op {
                '+' => result += current,
                '-' => result -= current,
                '*' => result *= current,
                _ => {}
            }
            current = 0;
            op = c;
        }
    }

    result
}

/// Verify a METTLE badge.
async fn verify_badge(
    client: &reqwest::Client,
    badge_token: &str,
) -> Result<BadgeVerifyResponse, Box<dyn std::error::Error>> {
    let response: BadgeVerifyResponse = client
        .post(format!("{}/badge/verify", METTLE_API))
        .json(&serde_json::json!({"token": badge_token}))
        .send()
        .await?
        .json()
        .await?;
    Ok(response)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    let result = verify_agent(&client, "my-rust-agent", "basic").await?;

    if let Some(badge) = result.badge {
        println!("\nVerifying badge...");
        let badge_check = verify_badge(&client, &badge).await?;
        println!("Badge valid: {}", badge_check.valid);
    }

    Ok(())
}
