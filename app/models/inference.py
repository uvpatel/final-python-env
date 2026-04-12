"""Dataclasses shared by the inference runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_BENCHMARK_NAME = "python_code_review_env"


@dataclass(slots=True)
class InferenceConfig:
    """Runtime configuration loaded from environment variables."""

    api_base_url: str
    model_name: str
    hf_token: str
    benchmark_name: str = DEFAULT_BENCHMARK_NAME
    request_timeout_s: float = 12.0
    max_retries: int = 2
    max_episode_steps: int = 12
    success_threshold: float = 0.94

    @classmethod
    def from_env(cls) -> "InferenceConfig":
        return cls(
            api_base_url=str(os.getenv("API_BASE_URL") or DEFAULT_API_BASE_URL),
            model_name=str(os.getenv("MODEL_NAME") or DEFAULT_MODEL_NAME),
            hf_token=str(os.getenv("HF_TOKEN") or ""),
            benchmark_name=str(os.getenv("OPENENV_BENCHMARK") or DEFAULT_BENCHMARK_NAME),
        )


@dataclass(slots=True)
class AgentDecision:
    """Validated action chosen for the next environment step."""

    action_type: str
    code: str | None = None
    source: str = "deterministic"
    error: str | None = None
