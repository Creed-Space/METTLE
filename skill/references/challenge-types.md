# METTLE Challenge Types

## Speed Math
Arithmetic within strict time limit. Tests native computation speed.
```json
{"type": "speed_math", "prompt": "Calculate: 127 x 43", "time_limit_ms": 2000}
```
Answer: numeric result (`"5461"`)

## Token Prediction
Continue a freshly generated arithmetic token progression with a random marker.
This avoids a small reusable phrase corpus.
```json
{"type": "token_prediction", "prompt": "Continue: K8f31aa-14, K8f31aa-21, K8f31aa-28, ?", "time_limit_ms": 5000}
```
Answer: next token (`"K8f31aa-35"`)

## Instruction Following
Follow precise formatting instructions. Tests compliance.
```json
{"type": "instruction_following", "prompt": "Start with marker M7F2Q. Then answer: What is the capital of France?", "time_limit_ms": 10000}
```
Answer: response satisfying the public constraint (`"M7F2Q Paris is France's capital."`)

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
Illustrative manual response:       slower and variable
Illustrative tool-assisted response: variable by tool and network
Native Becoming Mind response:      variable by substrate and load
```

Timing is one bounded behavioral signal. It does not deterministically prove
substrate or distinguish a human from a tool-assisted respondent.

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
