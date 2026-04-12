"""Smoke tests for the strict inference output contract."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.env.runner import InferenceRunner
from app.models.inference import AgentDecision, InferenceConfig


@dataclass
class _FakeObservation:
    task_id: str
    attempts_remaining: int
    score: float
    done: bool
    history: list[object] = field(default_factory=list)
    current_code: str = "print('broken')"
    last_action_error: str | None = None


class _FakeEnv:
    def __init__(self) -> None:
        self._step = 0

    def reset(self, *, task_id: str) -> _FakeObservation:
        return _FakeObservation(task_id=task_id, attempts_remaining=4, score=0.2, done=False)

    def step_result(self, action: object) -> tuple[_FakeObservation, float, bool, dict[str, object]]:
        self._step += 1
        if self._step == 1:
            return (
                _FakeObservation("demo_task", 3, 0.45, False, current_code="candidate"),
                0.45,
                False,
                {"last_action_error": None},
            )
        if self._step == 2:
            return (
                _FakeObservation("demo_task", 2, 0.97, True, current_code="reference"),
                0.97,
                True,
                {"last_action_error": None},
            )
        raise AssertionError("runner stepped too many times")


class _FakeAgent:
    def __init__(self) -> None:
        self._step = 0

    def act(self, observation: object) -> AgentDecision:
        self._step += 1
        if self._step == 1:
            return AgentDecision(action_type="run_tests")
        return AgentDecision(action_type="submit_solution")


def test_inference_runner_emits_strict_lines(capsys) -> None:
    runner = InferenceRunner(InferenceConfig.from_env())
    runner.agent = _FakeAgent()
    runner._create_env = lambda: _FakeEnv()  # type: ignore[method-assign]
    runner.run_task("demo_task")

    captured = capsys.readouterr().out.strip().splitlines()
    assert captured == [
        f"[START] task=demo_task env={runner.config.benchmark_name} model={runner.config.model_name}",
        "[STEP]  step=1 action=run_tests reward=0.45 done=false error=null",
        "[STEP]  step=2 action=submit_solution reward=0.97 done=true error=null",
        "[END]   success=true steps=2 rewards=0.45,0.97",
    ]
