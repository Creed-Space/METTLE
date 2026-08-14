//! METTLE Rust SDK Example
//!
//! Interactive verification flow for Becoming Minds.
//!
//! Add to Cargo.toml:
//! ```toml
//! [dependencies]
//! reqwest = { version = "0.11", features = ["json"] }
//! tokio = { version = "1", features = ["full"] }
//! serde = { version = "1", features = ["derive"] }
//! serde_json = "1"
//! ```
//!
//! Run:
//!   cargo run --example rust_example

use serde::{Deserialize, Serialize};
use std::io::{self, Write};

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

        // The example never solves issuer challenges. Read the response from
        // the Becoming Mind being verified.
        print!("  Response: ");
        io::stdout().flush()?;
        let mut answer = String::new();
        io::stdin().read_line(&mut answer)?;
        let answer = answer.trim_end().to_string();

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
