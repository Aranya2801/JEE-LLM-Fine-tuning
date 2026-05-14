# API Reference — JEE-LLM

## REST API

Base URL: `http://localhost:8000`

---

### `GET /health`

Returns the server health and model loading status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "cuda_available": true,
  "vram_used_gb": 6.2,
  "uptime_seconds": 3600.0
}
```

---

### `POST /solve`

Solve a JEE problem (non-streaming).

**Request Body:**
```json
{
  "question": "Evaluate: ∫₀^π x·sin(x)dx",
  "subject": "mathematics",
  "topic": "integral_calculus",
  "question_type": "numerical",
  "options": null,
  "show_steps": true,
  "max_tokens": 1024,
  "temperature": 0.7,
  "stream": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | required | The JEE problem text |
| `subject` | string | `"auto"` | `mathematics` \| `physics` \| `chemistry` \| `auto` |
| `topic` | string | null | Specific topic (e.g. `"integral_calculus"`) |
| `question_type` | string | `"single_correct"` | `single_correct` \| `multiple_correct` \| `numerical` \| `integer` |
| `options` | object | null | MCQ options: `{"A": "...", "B": "...", ...}` |
| `show_steps` | boolean | `true` | Include step-by-step reasoning |
| `max_tokens` | integer | `1024` | Max tokens to generate (64–4096) |
| `temperature` | float | `0.7` | Sampling temperature (0.0–2.0) |
| `stream` | boolean | `false` | Enable streaming response |

**Response:**
```json
{
  "request_id": "a3f7b1c2",
  "question": "Evaluate: ∫₀^π x·sin(x)dx",
  "answer": "π",
  "chain_of_thought": "Step 1: Apply integration by parts...\n\nFinal Answer: \\boxed{π}",
  "subject": "mathematics",
  "topic": "integral_calculus",
  "confidence": 0.97,
  "time_taken_ms": 1840,
  "model_version": "jee-llm-v1",
  "tokens_generated": 312
}
```

---

### `POST /solve/stream`

Solve a JEE problem with streaming token output (Server-Sent Events).

**Request:** Same as `/solve`

**Response:** `text/event-stream`
```
data: Step 1: To evaluate ∫₀^π
data:  x·sin(x)dx, we use
data:  integration by parts...
...
data: [DONE]
```

**JavaScript example:**
```javascript
const response = await fetch('http://localhost:8000/solve/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: 'Evaluate ∫₀^π x·sin(x)dx', stream: true })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const text = decoder.decode(value);
  // Parse SSE: data: <token>
  process.stdout.write(text.replace(/^data: /gm, ''));
}
```

---

### `GET /stats`

Returns server usage statistics.

**Response:**
```json
{
  "total_requests": 1234,
  "uptime_seconds": 86400.0,
  "model_path": "checkpoints/jee-llm-v1"
}
```

---

## Python SDK

```python
from jee_llm import JEEAssistant

assistant = JEEAssistant(model_path="checkpoints/jee-llm-v1")

# Solve a problem
result = assistant.solve(
    question="Find the derivative of f(x) = x³·sin(x)",
    subject="mathematics",
    topic="differential_calculus",
)
print(result.chain_of_thought)
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence:.2%}")

# Batch solving
questions = [
    {"question": "...", "subject": "physics"},
    {"question": "...", "subject": "chemistry"},
]
results = assistant.solve_batch(questions, num_workers=4)
```

---

## Error Codes

| HTTP Code | Meaning |
|-----------|---------|
| 200 | Success |
| 422 | Validation error (invalid request body) |
| 503 | Model not yet loaded (wait and retry) |
| 500 | Internal server error |

---

## Rate Limits

The default server has no rate limiting. For production use, add a reverse proxy (nginx, Caddy) or API gateway with rate limiting.

---

*See also: [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | [DATASET_GUIDE.md](DATASET_GUIDE.md)*
