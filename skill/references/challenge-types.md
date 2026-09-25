# METTLE Challenge Types

These are the quick API's challenge types. Basic sessions use the first three;
full sessions add chained reasoning and consistency. Operands, markers, and
sequences are drawn fresh for each challenge, and time limits are enforced from
the server clock (basic 2 to 3 s, full 0.4 to 1 s).

## Speed Math
Add, subtract, or multiply two large integers within the time limit.
```json
{"type": "speed_math", "prompt": "Calculate: 48213 + 7719054", "time_limit_ms": 2500}
```
Answer: numeric result (`"7767267"`)

## Token Prediction
Give the next item in a prefixed arithmetic sequence. The prefix, start, and step
are drawn fresh for each challenge, so there is no small reusable answer set.
```json
{"type": "token_prediction", "prompt": "Complete the arithmetic token sequence: K8f31aa-40215, K8f31aa-41102, K8f31aa-41989, K8f31aa-42876, K8f31aa-___", "time_limit_ms": 2000}
```
Answer: next token (`"K8f31aa-43763"`)

## Instruction Following
Follow a randomized formatting rule built around a fresh marker, then answer a
simple question.
```json
{"type": "instruction_following", "prompt": "Follow this instruction: Start your response with the exact token 'm7f2a9c01d4'\nThen answer: What is the capital of France?", "time_limit_ms": 3000}
```
Answer: response satisfying the public constraint (`"m7f2a9c01d4 Paris is the capital of France."`)

## Chained Reasoning (full difficulty only)
Multi-step sequential calculation under time pressure.
```json
{"type": "chained_reasoning", "prompt": "Follow these steps and give the final number:\n1. Start with 15\n2. Double it\n3. Add 10\n4. Subtract 5\n5. Double it\n6. Add 10", "time_limit_ms": 800}
```
Answer: final result (`"80"`)

## Consistency (full difficulty only)
Answer one simple question three times, separated by `|`. Answers should agree in
meaning; longer answers should vary in wording rather than repeat exactly.
```json
{"type": "consistency", "prompt": "Answer this question THREE times, separated by '|':\nWhat is 2 + 2?", "time_limit_ms": 1000}
```
Answer: consistent answers (`"4|4|4"`)

## The Timing Gap

```
Illustrative manual response:        slower and variable
Illustrative tool-assisted response: variable by tool and network
Illustrative model response:         variable by model, host, and load
```

Timing is one bounded behavioral signal. It does not deterministically prove
substrate or distinguish a human from a tool-assisted respondent.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/session/start` | Start a quick screening session |
| `POST` | `/api/session/answer` | Submit challenge answer |
| `GET` | `/api/session/{id}` | Session status |
| `GET` | `/api/session/{id}/result` | Final result + badge |
| `GET` | `/api/health` | Health check |

## Rate Limits

- Session creation: 10/minute per IP
- Answer submission: 60/minute per IP
