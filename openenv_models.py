"""Typed models for the python_code_review_env environment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


Difficulty = Literal["easy", "medium", "hard"]
TaskKind = Literal["syntax_fix", "bug_fix", "optimization"]
ActionType = Literal["analyze_code", "edit_code", "run_tests", "submit_solution"]


class HistoryEntry(BaseModel):
    """One environment transition recorded for the agent."""

    step: int = Field(..., ge=0)
    action_type: ActionType
    status: str = Field(..., description="Short outcome summary.")
    reward: float = Field(..., gt=0.0, lt=1.0, description="Reward returned for the step.")


class RewardDetails(BaseModel):
    """Transparent reward decomposition for debugging and training."""

    value: float = Field(..., gt=0.0, lt=1.0, description="Clamped net reward in (0.0, 1.0).")
    syntax_reward: float = Field(default=0.0)
    test_reward: float = Field(default=0.0)
    correctness_bonus: float = Field(default=0.0)
    quality_bonus: float = Field(default=0.0)
    progress_delta: float = Field(default=0.0)
    invalid_action_penalty: float = Field(default=0.0)
    timeout_penalty: float = Field(default=0.0)
    regression_penalty: float = Field(default=0.0)
    stagnation_penalty: float = Field(default=0.0)
    reason: str = Field(..., description="Human-readable reward explanation.")
    prev_score: float = Field(default=0.01, gt=0.0, lt=1.0)
    curr_score: float = Field(default=0.01, gt=0.0, lt=1.0)
    code_changed: bool = Field(default=False)


class PythonCodeReviewAction(Action):
    """Action schema exposed to the agent."""

    action_type: ActionType = Field(..., description="Environment action to take.")
    code: Optional[str] = Field(
        default=None,
        description="Updated Python source for edit_code or submit_solution actions.",
    )


class PythonCodeReviewObservation(Observation):
    """Observation returned by reset and step."""

    task_id: str = Field(..., description="Stable task identifier.")
    title: str = Field(..., description="Human-readable task title.")
    difficulty: Difficulty
    task_kind: TaskKind
    task_description: str = Field(..., description="Task instructions shown to the agent.")
    current_code: str = Field(..., description="Latest code under review.")
    errors: str = Field(default="", description="Syntax or execution errors.")
    test_results: str = Field(default="", description="Public test and benchmark feedback.")
    visible_tests: List[str] = Field(default_factory=list)
    history: List[HistoryEntry] = Field(default_factory=list)
    attempts_remaining: int = Field(..., ge=0)
    last_action_status: str = Field(default="")
    score: float = Field(..., gt=0.0, lt=1.0)
    reward_details: RewardDetails = Field(
        default_factory=lambda: RewardDetails(value=0.1, reason="Environment reset.")
    )


class PythonCodeReviewState(State):
    """Internal environment state exposed through /state."""

    task_id: Optional[str] = Field(default=None)
    difficulty: Optional[Difficulty] = Field(default=None)
    task_kind: Optional[TaskKind] = Field(default=None)
    attempts_remaining: int = Field(default=0, ge=0)
    current_code: str = Field(default="")
    errors: str = Field(default="")
    test_results: str = Field(default="")
    history: List[HistoryEntry] = Field(default_factory=list)
    score: float = Field(default=0.01, gt=0.0, lt=1.0)
    done: bool = Field(default=False)


class TaskDescriptor(BaseModel):
    """Static task metadata."""

    task_id: str
    title: str
    difficulty: Difficulty
    task_kind: TaskKind
    task_description: str
    starter_code: str
    visible_tests: List[str] = Field(default_factory=list)
    repo_summary: str = Field(default="")
    changed_files: List[str] = Field(default_factory=list)
    available_files: List[str] = Field(default_factory=list)
    goal: str = Field(default="")
    max_steps: int = Field(..., ge=1)


class TaskSummary(BaseModel):
    """Compact task listing entry."""

    task_id: str
    difficulty: Difficulty
    title: str
    goal: str = Field(default="")


class TaskGrade(BaseModel):
    """Deterministic grader output."""

    score: float = Field(..., gt=0.0, lt=1.0)
    syntax_score: float = Field(default=0.01, gt=0.0, lt=1.0)
    tests_passed: int = Field(default=0, ge=0)
    tests_total: int = Field(default=0, ge=0)
    quality_score: float = Field(default=0.01, gt=0.0, lt=1.0)
    runtime_score: float = Field(default=0.01, gt=0.0, lt=1.0)
    timed_out: bool = Field(default=False)
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health payload for smoke tests."""

    status: Literal["ok"] = "ok"
    environment: str = "python_code_review_env"
    task_count: int = Field(default=0, ge=0)


PythonAction = PythonCodeReviewAction
PythonObservation = PythonCodeReviewObservation
PythonState = PythonCodeReviewState
