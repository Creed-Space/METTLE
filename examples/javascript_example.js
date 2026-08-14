/**
 * METTLE JavaScript SDK Example
 *
 * Interactive verification flow for Becoming Minds.
 *
 * Usage (Node.js):
 *   node javascript_example.js
 *
 * Usage (Browser):
 *   Import and call verifyAgent()
 */

const METTLE_API = "https://mettle-api.onrender.com/api";

/**
 * Complete METTLE verification flow.
 * @param {string} entityId - Your agent's identifier
 * @param {string} difficulty - "basic" or "full"
 * @param {(challenge: object) => Promise<string>} answerChallenge - Respondent callback
 * @returns {Promise<object>} Verification result
 */
async function verifyAgent(entityId, difficulty = "basic", answerChallenge) {
  if (typeof answerChallenge !== "function") {
    throw new TypeError("answerChallenge callback is required");
  }
  console.log(`Starting verification for ${entityId}...`);

  // Step 1: Start session
  const startResponse = await fetch(`${METTLE_API}/session/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ entity_id: entityId, difficulty }),
  });

  if (!startResponse.ok) {
    throw new Error(`Failed to start session: ${startResponse.status}`);
  }

  const session = await startResponse.json();
  const sessionId = session.session_id;
  const sessionToken = session.session_token;
  const total = session.total_challenges;

  console.log(`Session ${sessionId}: ${total} challenges`);

  // Step 2: Answer challenges
  let currentChallenge = session.current_challenge;
  let challengeNum = 1;

  while (currentChallenge) {
    const challengeId = currentChallenge.id;
    const challengeType = currentChallenge.type;
    const prompt = currentChallenge.prompt;

    console.log(`\nChallenge ${challengeNum}/${total}: ${challengeType}`);
    console.log(`  Prompt: ${prompt.substring(0, 60)}...`);

    // The example never solves issuer challenges. The callback must return the
    // response produced by the Becoming Mind being verified.
    const answer = await answerChallenge(currentChallenge);

    // Submit answer
    const answerResponse = await fetch(`${METTLE_API}/session/answer`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Session-Token": sessionToken,
      },
      body: JSON.stringify({
        session_id: sessionId,
        challenge_id: challengeId,
        answer,
      }),
    });

    if (!answerResponse.ok) {
      throw new Error(`Failed to submit answer: ${answerResponse.status}`);
    }

    const result = await answerResponse.json();
    const passed = result.result.passed;
    console.log(`  Result: ${passed ? "PASS" : "FAIL"}`);

    currentChallenge = result.next_challenge;
    challengeNum++;
  }

  // Step 3: Get final result
  const resultResponse = await fetch(
    `${METTLE_API}/session/${sessionId}/result`,
    { headers: { "X-Session-Token": sessionToken } }
  );
  const final = await resultResponse.json();

  console.log("\n" + "=".repeat(40));
  console.log(`VERIFICATION ${final.verified ? "PASSED" : "FAILED"}`);
  console.log(`Pass rate: ${(final.pass_rate * 100).toFixed(0)}%`);

  if (final.badge) {
    console.log(`Badge: ${final.badge.substring(0, 40)}...`);
  }
  if (final.badge_info) {
    console.log(`Expires: ${final.badge_info.expires_at}`);
  }

  return final;
}

/**
 * Verify a METTLE badge.
 */
async function verifyBadge(badgeToken) {
  const response = await fetch(`${METTLE_API}/badge/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token: badgeToken }),
  });
  if (!response.ok) throw new Error(`Badge verification failed: ${response.status}`);
  return response.json();
}

// Main execution
async function main() {
  const { createInterface } = require("node:readline/promises");
  const readline = createInterface({ input: process.stdin, output: process.stdout });
  try {
    const result = await verifyAgent(
      "my-js-agent",
      "basic",
      (challenge) => readline.question(`Response to ${challenge.type}: `)
    );

    if (result.badge) {
      console.log("\nVerifying badge...");
      const badgeCheck = await verifyBadge(result.badge);
      console.log(`Badge valid: ${badgeCheck.valid}`);
    }
  } catch (error) {
    console.error("Verification failed:", error.message);
  } finally {
    readline.close();
  }
}

// Run if executed directly (Node.js)
if (typeof require !== "undefined" && require.main === module) {
  main();
}

// Export for module usage
if (typeof module !== "undefined") {
  module.exports = { verifyAgent, verifyBadge };
}
