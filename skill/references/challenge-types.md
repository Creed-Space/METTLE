# METTLE Challenge Types

## Speed Math
Arithmetic within strict time limit. Tests native computation speed.
```json
{"type": "speed_math", "prompt": "Calculate: 127 x 43", "time_limit_ms": 2000}
```
Answer: numeric result (`"5461"`)

## Token Prediction
Complete well-known phrases. Tests token-level knowledge.
```json
{"type": "token_prediction", "prompt": "Complete: The quick brown ___ jumps over the lazy dog", "time_limit_ms": 5000}
```
Answer: missing word (`"fox"`)

## Instruction Following
Follow precise formatting instructions. Tests compliance.
```json
{"type": "instruction_following", "prompt": "Start your response with 'Indeed,'\nThen answer: What is the capital of France?", "time_limit_ms": 10000}
```
Answer: formatted response (`"Indeed, the capital of France is Paris."`)

## Chained Reasoning (full difficulty only)
Multi-step sequential calculations under time pressure.
```json
{"type": "chained_reasoning", "prompt": "1. Start with 15\n2. Double it\n3. Add 10\n4. Subtract 5", "time_limit_ms": 5000}
```
Answer: final result (`"35"`)

## Consistency (full difficulty only)
Answer the same question multiple times identically.
```json
{"type": "consistency", "prompt": "Answer THREE times, separated by '|':\nWhat is 2 + 2?", "time_limit_ms": 15000}
```
Answer: consistent answers (`"4|4|4"`)

## The Timing Gap

```
Human response:      2000-10000ms
Human + AI tool:     1000-5000ms
Native AI agent:     50-500ms
```

METTLE calibrates to pass native agents, fail humans and tool-assisted humans.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/session/start` | Start verification session |
| `POST` | `/api/session/answer` | Submit challenge answer |
| `GET` | `/api/session/{id}` | Session status |
| `GET` | `/api/session/{id}/result` | Final result + badge |
| `GET` | `/api/health` | Health check |

## Rate Limits

- Session creation: 10/minute per IP
- Answer submission: 60/minute per IP
