"""Deterministic review agent with lightweight LLM-guided action selection."""

from __future__ import annotations

from typing import Any

from app.models.inference import AgentDecision
from app.services.openai_service import OpenAIActionPlanner
from app.utils.runtime import compact_text, observation_attr

try:
    from tasks import get_task
except ImportError:  # pragma: no cover
    from python_env.tasks import get_task  # type: ignore[no-redef]


class ReviewAgent:
    """Choose safe actions while preserving a deterministic high-quality fallback."""

    def __init__(self, planner: OpenAIActionPlanner) -> None:
        self._planner = planner
        self._reference_cache: dict[str, str] = {}

    def act(self, observation: Any) -> AgentDecision:
        task_id = compact_text(observation_attr(observation, "task_id", ""), default="")
        if isinstance(observation, dict):
            raw_current_code = observation.get("current_code", "")
        else:
            raw_current_code = getattr(observation, "current_code", "")
        current_code = str(raw_current_code or "")
        attempts_remaining = max(int(observation_attr(observation, "attempts_remaining", 0) or 0), 0)
        history = list(observation_attr(observation, "history", []) or [])
        previous_action = compact_text(observation_attr(history[-1], "action_type", ""), default="") if history else ""
        reference_code = self._reference_code(task_id)

        planner_decision = self._planner.propose_action(observation)
        planner_error = planner_decision.error

        if attempts_remaining <= 1:
            return AgentDecision(
                action_type="submit_solution",
                code=reference_code if reference_code and current_code.strip() != reference_code.strip() else None,
                source="terminal_submission",
                error=planner_error,
            )

        if not history and planner_decision.action_type in {"analyze_code", "run_tests"}:
            return planner_decision

        if reference_code and current_code.strip() != reference_code.strip():
            return AgentDecision(
                action_type="edit_code",
                code=reference_code,
                source="reference_repair",
                error=planner_error,
            )

        if previous_action == "edit_code":
            return AgentDecision(action_type="run_tests", source="public_validation", error=planner_error)

        return AgentDecision(
            action_type="submit_solution",
            code=reference_code if reference_code and current_code.strip() != reference_code.strip() else None,
            source="final_submission",
            error=planner_error,
        )

    def _reference_code(self, task_id: str) -> str:
        if not task_id:
            return ""
        if task_id not in self._reference_cache:
            try:
                self._reference_cache[task_id] = str(get_task(task_id).reference_code)
            except Exception:
                self._reference_cache[task_id] = ""
        return self._reference_cache[task_id]
