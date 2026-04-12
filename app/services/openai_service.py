"""OpenAI-compatible action planner backed by the Hugging Face router."""

from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI

from app.models.inference import AgentDecision, InferenceConfig
from app.utils.runtime import compact_text, observation_attr, suppress_output


ALLOWED_ACTIONS = {"analyze_code", "edit_code", "run_tests", "submit_solution"}


class OpenAIActionPlanner:
    """Ask an OpenAI-compatible model for the next safe environment action."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.client = OpenAI(base_url=config.api_base_url, api_key=config.hf_token) if config.hf_token else None

    def propose_action(self, observation: Any) -> AgentDecision:
        if self.client is None:
            return AgentDecision(action_type="run_tests", source="fallback", error="HF_TOKEN missing")

        prompt = self._build_prompt(observation)
        for attempt in range(self.config.max_retries + 1):
            try:
                with suppress_output():
                    response = self.client.chat.completions.create(
                        model=self.config.model_name,
                        temperature=0,
                        max_tokens=120,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a deterministic OpenEnv controller. "
                                    "Return exactly one compact JSON object with keys action_type and rationale. "
                                    "Allowed action_type values: analyze_code, run_tests, submit_solution. "
                                    "Never emit markdown."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        response_format={"type": "json_object"},
                    )
                message = response.choices[0].message.content or ""
                return self._parse_action(message)
            except Exception as exc:
                if attempt >= self.config.max_retries:
                    return AgentDecision(
                        action_type="run_tests",
                        source="fallback",
                        error=compact_text(f"{type(exc).__name__}: {exc}", default="LLM failure"),
                    )
                time.sleep(0.2 * (attempt + 1))

        return AgentDecision(action_type="run_tests", source="fallback", error="LLM retries exhausted")

    def _build_prompt(self, observation: Any) -> str:
        return (
            f"Task ID: {compact_text(observation_attr(observation, 'task_id', ''), default='unknown')}\n"
            f"Description: {compact_text(observation_attr(observation, 'task_description', ''), default='none', limit=400)}\n"
            f"Current score: {float(observation_attr(observation, 'score', 0.01) or 0.01):.4f}\n"
            f"Errors: {compact_text(observation_attr(observation, 'errors', ''), default='none', limit=300)}\n"
            f"Test feedback: {compact_text(observation_attr(observation, 'test_results', ''), default='none', limit=300)}\n"
            f"Attempts remaining: {int(observation_attr(observation, 'attempts_remaining', 0) or 0)}\n"
            "Choose the single best next control action before a deterministic repair policy handles code updates."
        )

    def _parse_action(self, content: str) -> AgentDecision:
        try:
            payload = json.loads(content)
        except Exception:
            return AgentDecision(action_type="run_tests", source="fallback", error="invalid LLM payload")

        action_type = compact_text(payload.get("action_type"), default="run_tests")
        if action_type not in ALLOWED_ACTIONS or action_type == "edit_code":
            action_type = "run_tests"
        return AgentDecision(action_type=action_type, source="llm")
