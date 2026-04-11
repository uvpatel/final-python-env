---
title: TorchReview Copilot
emoji: 🧠
colorFrom: orange
colorTo: red
sdk: docker
pinned: false
app_port: 8000
tags:
  - pytorch
  - gradio
  - fastapi
  - openenv
  - code-review
---

# TorchReview Copilot

TorchReview Copilot is an **AI-powered code review and improvement system using PyTorch** to analyze Python code, predict quality, generate structured improvement suggestions, and compute an RL-ready reward score.

It upgrades the original OpenEnv hackathon environment into a judge-friendly product demo: a polished Hugging Face Space on top, with the deterministic OpenEnv validation engine still preserved underneath.

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/uvpatel7271/final-python-env)  
**Repository:** [uvpatel/final-python-env](https://github.com/uvpatel/final-python-env)

## Problem Statement

Engineering teams lose time during incident response and code review because broken Python snippets often arrive with noisy traces, partial test output, and unclear ownership. Before fixing anything, someone still has to answer:

- Is this a syntax issue, a logic bug, or a performance regression?
- How risky is the repair?
- What should be checked first?

That triage step is repetitive, error-prone, and often slows down the actual fix.

## Solution

TorchReview Copilot turns code, traceback text, and a short context window into a practical code-review report:

- **Issue classification:** syntax, logic, or performance
- **ML quality score:** predicted code quality from PyTorch embeddings
- **Reward score:** RL-ready score from model quality, lint quality, and complexity penalty
- **Live Triage Radar:** confidence visualization for all issue classes
- **Nearest known pattern:** the closest OpenEnv task match
- **Improvement plan:** step 1 syntax/bug fixes, step 2 edge cases, step 3 scalability

The result is a demo that feels like a real AI debugging assistant rather than a backend-only environment.

## Why PyTorch Matters

This project uses **PyTorch for real inference**, not placeholder branching:

- `transformers` + `torch` load `huggingface/CodeBERTa-small-v1`
- the model encodes code snippets and failure context into embeddings
- embeddings are compared against curated OpenEnv issue prototypes
- the final decision blends model similarity with lightweight static analysis signals

That gives the demo an actual model-backed quality and issue scoring path while keeping it CPU-friendly for Hugging Face Spaces.

## How It Works

### Pipeline

`Input code + context window + traceback -> static checks -> PyTorch embeddings -> quality + issue prediction -> suggestion engine -> reward computation -> UI/API output`

### Detailed Flow

1. The user pastes Python code and optional traceback or benchmark output.
2. TorchReview extracts lightweight static signals:
   - parser success/failure
   - assertion-style test language
   - lint/style issues
   - nested-loop depth and complexity pressure
3. CodeBERTa runs through PyTorch to embed the combined input.
4. The embedding is compared against built-in issue prototypes derived from the OpenEnv task catalog and reference implementations.
5. The UI returns:
   - top issue label
   - confidence radar
   - repair risk
   - ML quality score
   - RL-ready reward score
   - nearest known bug pattern
   - three-step improvement plan

### Reward Formula

The current reward computation is:

```text
reward = (0.5 x ML_quality_score) + (0.3 x lint_score) - (0.2 x complexity_penalty)
```

This keeps the project compatible with OpenEnv-style reinforcement learning workflows.

## Built-In Demo Scenarios

The app ships with three grounded examples reused from the OpenEnv tasks:

1. **Syntax regression:** broken invoice normalization helper
2. **Logic bug:** session window boundary failure
3. **Performance bottleneck:** slow active-user ranking pipeline

These examples make the classification differences obvious during judging and video demos.

## Tech Stack

- **PyTorch** for embedding inference
- **Transformers** for `CodeBERTa-small-v1`
- **Gradio** for the polished Hugging Face Space UI
- **FastAPI** for the app server
- **OpenEnv** for deterministic validation endpoints and environment compatibility
- **Pydantic** for typed schemas

## Features

- PyTorch-powered code quality inference
- Static analysis for syntax, lint, and complexity
- Context-window-aware review flow
- RL-ready reward shaping
- Live Triage Radar visualization
- Three-step improvement plan:
  1. syntax checking and bug fixes
  2. edge-case handling
  3. scalability improvements

## Hugging Face Space UX

The root app now presents a production-style triage experience:

- a clear problem/solution hero section
- example scenario selector
- code and traceback inputs
- context window input
- **Live Triage Radar**
- structured improvement plan
- reward and quality score display
- visible model/backend notes

The underlying OpenEnv endpoints remain available for compatibility and evaluation.

## Screenshots

Add screenshots after deployment:

- `docs/screenshots/home.png` -> hero + inputs
- `docs/screenshots/triage-radar.png` -> confidence visualization
- `docs/screenshots/fix-plan.png` -> structured output panel

Suggested markdown once captured:

```md
![TorchReview Copilot Home](docs/screenshots/home.png)
![Live Triage Radar](docs/screenshots/triage-radar.png)
![Fix Plan Output](docs/screenshots/fix-plan.png)
```

## Local Setup

### 1. Install dependencies

```bash
pip install .
```

### 2. Run the application

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3. Open the demo

Visit:

```text
http://localhost:8000/
```

### 4. Verify OpenEnv compatibility

```bash
curl http://localhost:8000/health
curl http://localhost:8000/state
```

## Docker

```bash
docker build -t torchreview-copilot -f server/Dockerfile .
docker run --rm -p 8000:8000 torchreview-copilot
```

Expected checks:

```bash
curl http://localhost:8000/
curl http://localhost:8000/health
```

## Project Structure

```text
python_env/
├── client.py
├── graders/
├── server/
│   ├── app.py
│   ├── demo.py
│   └── env.py
├── tasks/
├── triage.py
├── triage_catalog.py
├── triage_models.py
├── inference.py
└── tests/
```

## OpenEnv Compatibility

The hackathon backend is still present:

- deterministic task grading
- structured action/observation/state models
- `/health`, `/state`, `/reset`, `/step`, and related environment routes

This means the product demo is not detached from evaluation; it is layered on top of the original OpenEnv system.

## Demo Script

See [DEMO_SCRIPT.md](DEMO_SCRIPT.md) for the 60-90 second recording flow.

Short version:

1. Open the Space and introduce the problem.
2. Load the syntax example.
3. Show the Live Triage Radar and issue label.
4. Explain the PyTorch embedding step.
5. Show the matched pattern and fix plan.
6. Show the reward score and explain how it can be used inside an RL environment.
7. Switch to the performance example to prove the model distinguishes issue classes.

## Limitations

- The classifier uses pretrained embeddings plus prototype similarity, not a custom fine-tuned model.
- First model load may take longer on a cold Hugging Face Space.
- The current demo focuses on short Python snippets rather than full multi-file repositories.

## Future Work

- fine-tune the PyTorch classifier on a larger bug triage dataset
- add repository-level file context and diff-aware analysis
- include automated patch suggestions after triage
- track remediation outcomes as a feedback loop for future ranking improvements
