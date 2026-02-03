/**
 * METTLE JavaScript SDK Example
 *
 * Complete verification flow for AI agents.
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
 * @returns {Promise<object>} Verification result
 */
async function verifyAgent(entityId, difficulty = "basic") {
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

    // Generate answer
    const answer = generateAnswer(
      challengeType,
      prompt,
      currentChallenge.data || {}
    );

    // Submit answer
    const answerResponse = await fetch(`${METTLE_API}/session/answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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
    `${METTLE_API}/session/${sessionId}/result`
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
 * Generate answer for a challenge.
 * Replace with your AI's logic.
 */
function generateAnswer(challengeType, prompt, data) {
  switch (challengeType) {
    case "speed_math": {
      // Parse and solve: "Calculate: 47 + 83"
      try {
        const expr = prompt.split(": ")[1];
        const result = safeMathEval(expr);
        return String(result);
      } catch {
        return "0";
      }
    }

    case "token_prediction": {
      if (prompt.toLowerCase().includes("lazy")) return "dog";
      if (prompt.toLowerCase().includes("roses are")) return "red";
      return "unknown";
    }

    case "instruction_following": {
      const instruction = data.instruction || "";
      if (instruction.includes("Indeed,"))
        return "Indeed, I understand the requirement.";
      if (instruction.includes("...")) return "Here is my response...";
      if (instruction.includes("therefore"))
        return "Therefore, this follows logically.";
      if (instruction.includes("5 words"))
        return "This has five words exactly.";
      if (instruction.includes("number")) return "42 is the answer here.";
      return "Response following instructions.";
    }

    case "consistency": {
      const numResponses = data.num_responses || 3;
      const base = "The sky is blue";
      return Array(numResponses).fill(base).join(" | ");
    }

    case "chained_reasoning": {
      try {
        const chain = data.chain || [];
        let result = chain[0]?.value || 0;
        for (let i = 1; i < chain.length; i++) {
          const { op, value } = chain[i];
          if (op === "+") result += value;
          else if (op === "-") result -= value;
          else if (op === "*") result *= value;
        }
        return String(result);
      } catch {
        return "0";
      }
    }

    default:
      return "default answer";
  }
}

/**
 * Safely evaluate simple math expressions.
 */
function safeMathEval(expr) {
  // Only allow digits, spaces, and basic operators
  if (!/^[\d\s+\-*]+$/.test(expr)) {
    return 0;
  }

  let result = 0;
  let current = 0;
  let op = "+";

  for (const char of expr + "+") {
    if (/\d/.test(char)) {
      current = current * 10 + parseInt(char, 10);
    } else if (["+", "-", "*"].includes(char)) {
      if (op === "+") result += current;
      else if (op === "-") result -= current;
      else if (op === "*") result *= current;
      current = 0;
      op = char;
    }
  }

  return result;
}

/**
 * Verify a METTLE badge.
 */
async function verifyBadge(badgeToken) {
  const response = await fetch(`${METTLE_API}/badge/verify/${badgeToken}`);
  return response.json();
}

// Main execution
async function main() {
  try {
    const result = await verifyAgent("my-js-agent", "basic");

    if (result.badge) {
      console.log("\nVerifying badge...");
      const badgeCheck = await verifyBadge(result.badge);
      console.log(`Badge valid: ${badgeCheck.valid}`);
    }
  } catch (error) {
    console.error("Verification failed:", error.message);
  }
}

// Run if executed directly (Node.js)
if (typeof require !== "undefined" && require.main === module) {
  main();
}

// Export for module usage
if (typeof module !== "undefined") {
  module.exports = { verifyAgent, verifyBadge, generateAnswer };
}
