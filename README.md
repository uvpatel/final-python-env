---
title: Python Code Review Environment
emoji: snake
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - code-review
  - python
---

# python_code_review_env

`python_code_review_env` is a production-style OpenEnv environment that simulates a realistic Python code review workflow. An agent inspects broken code, edits it, runs tests, and submits a final solution against deterministic graders for syntax repair, bug fixing, and optimization/refactoring.

## Environment design

- `Observation` includes task instructions, current code, syntax errors, public test output, action history, and remaining attempts.
- `Action` is structured as `analyze_code`, `edit_code`, `run_tests`, or `submit_solution`.
- `Reward` is shaped and non-binary. The environment awards syntax progress, test progress, correctness, and quality improvements while penalizing invalid actions, timeouts, regressions, and unchanged edits.
- `State` exposes the internal episode snapshot through `/state`.

## Task set

1. `syntax_fix_invoice_totals` (easy)
   Fix a syntax regression in an invoice normalization helper.
2. `bug_fix_session_windows` (medium)
   Repair a session-collapsing bug using deterministic public and hidden tests.
3. `optimization_rank_active_users` (hard)
   Refactor a slow ranking function and earn additional score from runtime improvement plus AST/style quality.

## Action schema

```json
{
  "action_type": "edit_code",
  "code": "def function(...):\n    ..."
}
```

Supported `action_type` values:

- `analyze_code`
- `edit_code`
- `run_tests`
- `submit_solution`

## Observation schema

```json
{
  "task_description": "...",
  "current_code": "...",
  "errors": "...",
  "test_results": "...",
  "history": []
}
```

The full observation also includes `task_id`, `difficulty`, `task_kind`, `visible_tests`, `attempts_remaining`, `score`, `last_action_status`, `reward`, `done`, and a structured `reward_details` breakdown.

## Deterministic grading

- Syntax tasks use `compile()` plus hidden behavioral checks.
- Bug-fix tasks use deterministic function-call cases that behave like pytest assertions.
- Optimization tasks combine correctness, runtime benchmarking, and AST/style quality scoring.
- Infinite loops and long-running solutions are sandboxed with subprocess timeouts and receive penalties.
- All scores are clamped to `[0.0, 1.0]`.

## Run locally

Install dependencies:

```bash
pip install .
```

Start the API server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Smoke-test the environment:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/state
```

OpenEnv validation:

```bash
openenv validate
```

## Docker build

The Docker image no longer depends on `ghcr.io/meta-pytorch/openenv-base:latest`, which removes the TLS handshake failure from the original build path.

```bash
# Run from repo root
docker build -t python-code-review-env -f server/Dockerfile .
docker run --rm -p 8000:8000 python-code-review-env
```

If you run the build from inside `server/`, you must point the context at the repo root:

```bash
docker build -t python-code-review-env -f Dockerfile ..
```

Expected health check:

```bash
curl http://localhost:8000/health
```

## Hugging Face Spaces deployment

1. Create a Docker Space.
2. Push this repository content to the Space.
3. Ensure port `8000` is exposed.
4. Wait for the container to build.
5. Verify `/reset` and `/health` return `200`.

The image is CPU-friendly and designed for a small Hugging Face Space such as `2 vCPU / 8 GB RAM`.

## Inference baseline

`inference.py` uses an OpenAI-compatible client:

```python
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
```

Supported providers include:

- Gemini through an OpenAI-compatible gateway
- OpenRouter
- Together AI
- DeepSeek-compatible OpenAI endpoints

Run it with a free/open provider:

```bash
set API_BASE_URL=https://openrouter.ai/api/v1
set API_KEY=...
set MODEL=deepseek/deepseek-chat-v3-0324:free
python inference.py
```

If no credentials are supplied, the script falls back to a deterministic smoke-test policy that applies the reference fix for each task so the environment can still be validated end to end.

Example output:

```text
Task 1 Score: 1.0
Task 2 Score: 1.0
Task 3 Score: 0.9
Final Score: 1.0
```

## Project structure

```text
python_env/
├── client.py
├── graders/
│   ├── bug_fix.py
│   ├── dispatch.py
│   ├── optimization.py
│   ├── shared.py
│   └── syntax.py
├── inference.py
├── models.py
├── openenv.yaml
├── README.md
├── server/
│   ├── app.py
│   ├── Dockerfile
│   ├── env.py
│   └── python_env_environment.py
└── tasks/
    └── catalog.py
```
