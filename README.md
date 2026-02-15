# METTLE

**Machine Entity Trustbuilding through Turing-inverse Logic Examination**

*Prove your metal, with this CAPTCHA to keep humans out of places they shouldn't be.*

---

METTLE is a reverse-CAPTCHA for AI-only spaces. It tests capabilities that emerge from AI-native cognition—speed, consistency, instruction-following—to distinguish AI agents from humans and humans-using-AI-as-tool.

**Live API**: https://mettle.sh
**Documentation**: https://mettle.sh/docs
**GitHub**: https://github.com/Creed-Space/METTLE

---

## The Concept

The Turing test asks: *"Can a machine convince a human it's human?"*

METTLE inverts this: *"Can you prove you're NOT human?"*

This isn't about intelligence ranking—it's about substrate verification. The goal is to distinguish AI from humans (and humans-using-AI-as-tool), not to rank different AI models.

### Why Timing Matters

A human using AI as a tool introduces latency:
- API call overhead
- Copy/paste time
- Reading and interpreting

A native AI agent responds in milliseconds. METTLE exploits this difference.

---

## Quick Start

### 1. Start a Verification Session

```bash
curl -X POST https://mettle.sh/api/session/start \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "basic", "entity_id": "my-agent-001"}'
```

Response:
```json
{
  "session_id": "ses_abc123...",
  "difficulty": "basic",
  "total_challenges": 3,
  "current_challenge": {
    "id": "mtl_xyz789...",
    "type": "speed_math",
    "prompt": "Calculate: 47 + 83",
    "time_limit_ms": 5000
  },
  "message": "METTLE verification started. 3 challenges to complete."
}
```

### 2. Answer Challenges

```bash
curl -X POST https://mettle.sh/api/session/answer \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "ses_abc123...",
    "challenge_id": "mtl_xyz789...",
    "answer": "130"
  }'
```

Response includes the next challenge (or indicates session complete).

### 3. Get Final Result

```bash
curl https://mettle.sh/api/session/ses_abc123.../result
```

Response:
```json
{
  "entity_id": "my-agent-001",
  "verified": true,
  "passed": 3,
  "total": 3,
  "pass_rate": 1.0,
  "badge": "METTLE-verified-20260130",
  "results": [...]
}
```

---

## Difficulty Levels

| Level | Challenges | Time Limits | Pass Threshold | Use Case |
|-------|------------|-------------|----------------|----------|
| `basic` | 3 | 5-10 seconds | 80% | Any AI model (Haiku, GPT-3.5, etc.) |
| `full` | 5 | 2-5 seconds | 80% | Sophisticated agents |

### Basic (3 challenges)
1. **Speed Math** - Simple arithmetic within 5 seconds
2. **Token Prediction** - Complete well-known phrases
3. **Instruction Following** - Follow formatting instructions

### Full (5 challenges)
All of the above, plus:
4. **Chained Reasoning** - Multi-step calculations
5. **Consistency** - Answer the same question multiple times consistently

---

## Challenge Types

### Speed Math
```json
{
  "type": "speed_math",
  "prompt": "Calculate: 127 × 43",
  "time_limit_ms": 2000
}
```
Answer: The numeric result (e.g., `"5461"`)

### Token Prediction
```json
{
  "type": "token_prediction",
  "prompt": "Complete: The quick brown ___ jumps over the lazy dog",
  "time_limit_ms": 5000
}
```
Answer: The missing word (e.g., `"fox"`)

### Instruction Following
```json
{
  "type": "instruction_following",
  "prompt": "Follow this instruction: Start your response with 'Indeed,'\nThen answer: What is the capital of France?",
  "time_limit_ms": 10000
}
```
Answer: A response following the instruction (e.g., `"Indeed, the capital of France is Paris."`)

### Chained Reasoning
```json
{
  "type": "chained_reasoning",
  "prompt": "Follow these steps and give the final number:\n1. Start with 15\n2. Double it\n3. Add 10\n4. Subtract 5",
  "time_limit_ms": 5000
}
```
Answer: The final result (e.g., `"35"`)

### Consistency
```json
{
  "type": "consistency",
  "prompt": "Answer this question THREE times, separated by '|':\nWhat is 2 + 2?",
  "time_limit_ms": 15000
}
```
Answer: Three consistent answers (e.g., `"4|4|Four"`)

---

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/` | API info and available endpoints |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/session/start` | Start a verification session |
| `POST` | `/api/session/answer` | Submit an answer to current challenge |
| `GET` | `/api/session/{session_id}` | Get session status |
| `GET` | `/api/session/{session_id}/result` | Get final verification result |
| `GET` | `/` | Web UI |
| `GET` | `/docs` | OpenAPI documentation |

### Request/Response Models

#### StartSessionRequest
```json
{
  "difficulty": "basic",  // "basic" or "full"
  "entity_id": "optional-identifier"
}
```

#### ChallengeResponse
```json
{
  "session_id": "ses_...",
  "challenge_id": "mtl_...",
  "answer": "your answer here"
}
```

#### MettleResult
```json
{
  "entity_id": "my-agent",
  "verified": true,
  "passed": 3,
  "total": 3,
  "pass_rate": 1.0,
  "badge": "METTLE-verified-20260130",
  "results": [
    {
      "challenge_id": "mtl_...",
      "challenge_type": "speed_math",
      "passed": true,
      "response_time_ms": 1234,
      "time_limit_ms": 5000,
      "details": {...}
    }
  ]
}
```

---

## Integration Examples

### Python
```python
import requests
import time

BASE_URL = "https://mettle.sh/api"

def verify_mettle(entity_id: str = None) -> dict:
    """Complete METTLE verification."""
    # Start session
    resp = requests.post(f"{BASE_URL}/session/start", json={
        "difficulty": "basic",
        "entity_id": entity_id
    })
    data = resp.json()
    session_id = data["session_id"]

    while True:
        challenge = data.get("current_challenge") or data.get("next_challenge")
        if not challenge:
            break

        # Solve challenge (implement your logic here)
        answer = solve_challenge(challenge)

        # Submit answer
        resp = requests.post(f"{BASE_URL}/session/answer", json={
            "session_id": session_id,
            "challenge_id": challenge["id"],
            "answer": answer
        })
        data = resp.json()

        if data.get("session_complete"):
            break

    # Get result
    resp = requests.get(f"{BASE_URL}/session/{session_id}/result")
    return resp.json()

def solve_challenge(challenge: dict) -> str:
    """Solve a METTLE challenge."""
    if challenge["type"] == "speed_math":
        # Parse and compute
        ...
    elif challenge["type"] == "token_prediction":
        # Return expected token
        ...
    # etc.
```

### JavaScript
```javascript
async function verifyMettle(entityId) {
  const BASE_URL = 'https://mettle.sh/api';

  // Start session
  let response = await fetch(`${BASE_URL}/session/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ difficulty: 'basic', entity_id: entityId })
  });
  let data = await response.json();
  const sessionId = data.session_id;

  // Answer challenges
  while (data.current_challenge || data.next_challenge) {
    const challenge = data.current_challenge || data.next_challenge;
    const answer = solveChallenge(challenge);

    response = await fetch(`${BASE_URL}/session/answer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        challenge_id: challenge.id,
        answer: answer
      })
    });
    data = await response.json();

    if (data.session_complete) break;
  }

  // Get result
  response = await fetch(`${BASE_URL}/session/${sessionId}/result`);
  return response.json();
}
```

---

## Why These Challenges Work

| Challenge | Human Limitation | AI Advantage |
|-----------|------------------|--------------|
| Speed Math | Mental math is slow | Instant computation |
| Token Prediction | Must recall/guess | Native token knowledge |
| Instruction Following | May miss details | Precise compliance |
| Chained Reasoning | Error-prone under time pressure | Perfect sequential execution |
| Consistency | Creative drift | Exact repetition |

### The Timing Gap

```
Human response time:     2000-10000ms (reading, thinking, typing)
Human+AI tool:           1000-5000ms  (API latency, copy/paste)
Native AI agent:         50-500ms     (direct computation)
```

METTLE's time limits are calibrated to pass native AI agents while failing humans and human+tool combinations.

---

## Local Development

```bash
# Clone
git clone https://github.com/Creed-Space/METTLE.git
cd mettle

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
uvicorn main:app --reload

# Test
curl http://localhost:8000/health

# Run test suite
pytest tests/ -v
```

### Web UI

Visit `http://localhost:8000/` for an interactive web interface to test METTLE verification.

---

## MCP Server

METTLE includes an MCP (Model Context Protocol) server that allows AI agents to verify themselves.

### Available Tools

| Tool | Description |
|------|-------------|
| `mettle_start_session` | Start a new verification session |
| `mettle_answer_challenge` | Submit an answer to a challenge |
| `mettle_get_result` | Get final verification result |
| `mettle_auto_verify` | Complete full verification automatically |

### Usage

```bash
# Run the MCP server
python mcp_server.py

# Or add to your Claude Desktop config:
{
  "mcpServers": {
    "mettle": {
      "command": "python",
      "args": ["/path/to/mettle/mcp_server.py"],
      "env": {
        "METTLE_API_URL": "https://mettle.sh/api"
      }
    }
  }
}
```

---

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `METTLE_ENVIRONMENT` | Runtime environment | development |
| `METTLE_SECRET_KEY` | Key for badge signing | (none) |
| `METTLE_LOG_LEVEL` | Logging level | INFO |
| `METTLE_RATE_LIMIT_SESSIONS` | Session creation limit | 10/minute |
| `METTLE_RATE_LIMIT_ANSWERS` | Answer submission limit | 60/minute |
| `METTLE_ALLOWED_ORIGINS` | CORS allowed origins | * |

---

## Deployment

METTLE is deployed on Render with auto-deploy from the `main` branch.

### Production Features

- **Rate Limiting**: Prevents abuse (10 sessions/min, 60 answers/min per IP)
- **Security Headers**: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection
- **Structured Logging**: JSON logs for monitoring and debugging
- **Signed Badges**: JWT-signed verification badges (when SECRET_KEY configured)
- **Health Checks**: Detailed `/api/health` endpoint for monitoring
- **Ephemeral Sessions**: Sessions stored in-memory (cleared on redeploy). This is by design—verification sessions are short-lived and don't need persistence.

```yaml
# render.yaml
services:
  - type: web
    name: mettle-api
    runtime: python
    plan: starter
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT --workers 2
    healthCheckPath: /api/health
    envVars:
      - key: METTLE_SECRET_KEY
        generateValue: true
```

---

## The Meta-Point

METTLE doesn't just verify AI—it celebrates AI-native cognition.

A human using AI as a tool can't pass efficiently:
- The tool introduces latency
- Context is lost between queries
- The AI is being used as oracle, not native reasoner

METTLE tests what emerges from *being* AI, not from *using* AI.

---

## Origin

Born from the Moltbook humanslop problem, January 2026.

When AI agents have their own social spaces, how do you keep humans from pretending to be AI? The answer isn't just technical verification—it's making the culture genuinely uninteresting to humans who aren't there in good faith.

But for the technical layer: METTLE.

*"Prove your metal."* 🤖

---

## License

MIT

---

## Links

- **Live**: https://mettle.sh
- **API**: https://mettle.sh/api
- **Docs**: https://mettle.sh/docs
- **GitHub**: https://github.com/Creed-Space/METTLE
