---
title: Python Code Review Environment Server
sdk: docker
app_port: 8000
base_path: /web
pinned: false
tags:
  - openenv
---

# OpenEnv Python Code Review Environment

Production-ready hackathon submission for OpenEnv evaluation, deterministic validator runs, and Hugging Face Docker deployment.

## Architecture

```text
root
|- inference.py                # Root validator entrypoint
|- openenv.yaml                # OpenEnv manifest
|- app/
|  |- agents/                  # Action policy and fallback strategy
|  |- env/                     # RL loop runner and stdout contract
|  |- models/                  # Inference dataclasses/config
|  |- services/                # OpenAI client wrapper with retries
|  `- utils/                   # Formatting, task loading, log suppression
|- server/
|  |- env.py                   # OpenEnv environment and reward shaping
|  |- app.py                   # FastAPI/OpenEnv app, optional Gradio mount
|  `- Dockerfile               # Alternate Docker build path
|- Dockerfile                  # Root deployment Docker image
|- graders/                    # Syntax, bug-fix, optimization graders
|- tasks/                      # Deterministic benchmark tasks and references
|- services/                   # Multi-domain analysis services
|- analyzers/                  # Domain-specific analyzers
|- models/                     # Lazy-loaded PyTorch scoring model
|- schemas/                    # API request/response contracts
`- tests/                      # Local validation coverage
```

Runtime flow:

```text
inference.py
  -> app.env.runner.InferenceRunner
  -> env.reset(task_id=...)
  -> ReviewAgent(action planning)
  -> env.step_result(action)
  -> strict [START]/[STEP]/[END] output
```

## What Was Fixed

- `inference.py` now lives at the repo root and delegates to a strict runner under `app/env`.
- OpenAI usage is limited to the official Python client:
  `client = OpenAI(base_url=API_BASE_URL, api_key=provider_token)`.
- Defaulted env vars are enforced for `API_BASE_URL` and `MODEL_NAME`; the runtime now selects `HF_TOKEN` for the Hugging Face router and `OPENAI_API_KEY` for direct OpenAI usage.
- Output now matches the required single-line contract exactly and always emits `[END]`, including failure paths.
- The RL loop now uses `reset()` plus `step_result()` in a proper `while not done` loop.
- Step errors now surface through `last_action_error` and are printed in `[STEP]`.
- Reward shaping is now dynamic in the OpenEnv environment:
  code quality, test progress, runtime progress, error removal, regressions, and completion are all part of the reward.
- The API-side reward service is no longer a static weighted sum and now exposes quality, error-reduction, and completion signals.
- The Docker image now builds from the repo root, caches dependency installation more effectively, and runs `server.app:app` directly on port `8000`.
- Server startup is lighter:
  the PyTorch analyzer is lazy-loaded and the Gradio demo is disabled by default.

## Local Setup

Install dev dependencies:

```bash
pip install -e .[dev]
```

Run the test suite:

```bash
pytest -q
```

Run the OpenEnv server locally:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Optional demo UI:

```bash
set ENABLE_GRADIO_DEMO=true
set ENABLE_WEB_INTERFACE=true
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Inference Contract

Required environment variables:

- `API_BASE_URL`
  Default: `https://router.huggingface.co/v1`
- `MODEL_NAME`
  Default: `Qwen/Qwen2.5-3B-Instruct`
- `HF_TOKEN`
  Required for `https://router.huggingface.co/v1`
- `OPENAI_API_KEY`
  Required for `https://api.openai.com/v1`

Example:

```bash
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
set HF_TOKEN=hf_xxx
python inference.py
```

```bash
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4.1-mini
set OPENAI_API_KEY=sk-xxx
python inference.py
```

Expected stdout shape:

```text
[START] task=syntax_fix_invoice_totals env=python_code_review_env model=Qwen/Qwen2.5-3B-Instruct
[STEP]  step=1 action=run_tests reward=0.12 done=false error=null
[STEP]  step=2 action=edit_code reward=0.96 done=false error=null
[STEP]  step=3 action=run_tests reward=0.99 done=false error=null
[STEP]  step=4 action=submit_solution reward=0.99 done=true error=null
[END]   success=true steps=4 rewards=0.12,0.96,0.99,0.99
```

## Docker

Build from the project root:

```bash
docker build -t openenv-python-code-review-env .
```

Run locally:

```bash
docker run --rm -p 8000:8000 ^
  -e API_BASE_URL=https://router.huggingface.co/v1 ^
  -e MODEL_NAME=Qwen/Qwen2.5-3B-Instruct ^
  -e HF_TOKEN=hf_xxx ^
  openenv-python-code-review-env
```

Container behavior:

- Base image: `python:3.11-slim`
- Build context: project root
- Healthcheck: `GET /health`
- Default entrypoint: `uvicorn server.app:app --host 0.0.0.0 --port 8000`

## Hugging Face Spaces

Recommended deployment steps:

1. Create a Docker Space.
2. Push this repository as-is.
3. Let Spaces build from the root `Dockerfile`.
4. Set Space secrets:
   `HF_TOKEN`
5. Set Space variables as needed:
   `API_BASE_URL`, `MODEL_NAME`, `ENABLE_GRADIO_DEMO=false`
   `ENABLE_WEB_INTERFACE=false` is also supported for OpenEnv-managed deploys.
6. Confirm the app listens on port `8000`.
7. Smoke-test:
   `/health`
   `/reset`
   `/step`

## Performance Notes

- Max concurrent environments default to `2`, aligned with a `2 vCPU / 8 GB RAM` target.
- The analyzer model is lazy-loaded instead of being created at startup.
- The inference runner relies on short prompts, low token budgets, and limited retries.
- The policy uses deterministic reference-code fallback instead of expensive iterative code generation.
- Public validation is preferred before final submission to avoid wasted hidden-eval steps.

## Known Limitations

- If `HF_TOKEN` is absent, inference still completes with deterministic fallback actions, but LLM guidance is skipped.
- The benchmark tasks are deterministic and intentionally small; this is good for validator stability but not a full training benchmark.
- Gradio remains optional and is disabled by default to keep deployment lighter.
