"""Dataclasses shared by the inference runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_BENCHMARK_NAME = "python_code_review_env"


def _resolve_api_key(api_base_url: str) -> str:
    """Choose the correct provider token for the configured endpoint."""

    normalized = api_base_url.strip().lower()
    hf_token = str(os.getenv("HF_TOKEN") or "").strip()
    openai_api_key = str(os.getenv("OPENAI_API_KEY") or "").strip()

    if "api.openai.com" in normalized:
        return openai_api_key or hf_token
    return hf_token or openai_api_key


@dataclass(slots=True)
class InferenceConfig:
    """Runtime configuration loaded from environment variables."""

    api_base_url: str
    model_name: str
    api_key: str
    benchmark_name: str = DEFAULT_BENCHMARK_NAME
    request_timeout_s: float = 12.0
    max_retries: int = 2
    max_episode_steps: int = 12
    success_threshold: float = 0.94

    @classmethod
    def from_env(cls) -> "InferenceConfig":
        api_base_url = str(os.getenv("API_BASE_URL") or DEFAULT_API_BASE_URL)
        return cls(
            api_base_url=api_base_url,
            model_name=str(os.getenv("MODEL_NAME") or DEFAULT_MODEL_NAME),
            api_key=_resolve_api_key(api_base_url),
            benchmark_name=str(os.getenv("OPENENV_BENCHMARK") or DEFAULT_BENCHMARK_NAME),
        )


@dataclass(slots=True)
class AgentDecision:
    """Validated action chosen for the next environment step."""

    action_type: str
    code: str | None = None
    source: str = "deterministic"
    error: str | None = None
