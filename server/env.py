"""OpenEnv environment implementation for Python code review tasks."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..graders import grade_task
    from ..graders.shared import component_score, safe_ratio, strict_score
    from ..openenv_models import (
        HistoryEntry,
        PythonCodeReviewAction,
        PythonCodeReviewObservation,
        PythonCodeReviewState,
        RewardDetails,
        TaskGrade,
    )
    from ..tasks import ReviewTask, list_tasks, select_task
except ImportError:
    from graders import grade_task
    from graders.shared import component_score, safe_ratio, strict_score
    from openenv_models import (
        HistoryEntry,
        PythonCodeReviewAction,
        PythonCodeReviewObservation,
        PythonCodeReviewState,
        RewardDetails,
        TaskGrade,
    )
    from tasks import ReviewTask, list_tasks, select_task


def _empty_grade() -> TaskGrade:
    return TaskGrade(
        score=component_score(0.01),
        syntax_score=component_score(0.01),
        tests_passed=0,
        tests_total=0,
        quality_score=component_score(0.01),
        runtime_score=component_score(0.01),
    )


def _reward_value(value: float) -> float:
    return strict_score(value)


class PythonCodeReviewEnvironment(
    Environment[PythonCodeReviewAction, PythonCodeReviewObservation, PythonCodeReviewState]
):
    """Structured environment for deterministic Python code review workflows."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, verbose: bool = False, **_: Any) -> None:
        super().__init__()
        self.verbose = verbose
        self._task: ReviewTask = list_tasks()[0]
        self._current_code: str = self._task.starter_code
        self._history: list[HistoryEntry] = []
        self._last_reward = RewardDetails(value=0.1, reason="Environment initialized.")
        self._last_action_error: str | None = None
        self._current_grade = _empty_grade()
        self._state = PythonCodeReviewState(episode_id=str(uuid4()), step_count=0)
        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> PythonCodeReviewObservation:
        task_id = kwargs.get("task_id")
        self._task = select_task(seed=seed, task_id=task_id)
        self._current_code = self._task.starter_code
        self._history = []
        self._last_action_error = None
        self._last_reward = RewardDetails(value=0.1, reason="Environment reset.")
        self._current_grade, self._last_action_error = self._safe_grade_task(
            self._task,
            self._current_code,
            include_hidden=False,
        )

        self._state = PythonCodeReviewState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            task_kind=self._task.task_kind,
            attempts_remaining=self._task.max_steps,
            current_code=self._current_code,
            errors=self._format_errors(self._current_grade),
            test_results=self._format_test_results(self._current_grade),
            history=[],
            score=self._current_grade.score,
            done=False,
        )
        return self._build_observation(
            grade=self._current_grade,
            status=f"Loaded task {self._task.task_id}.",
            reward_details=self._last_reward,
        )

    def step(
        self,
        action: PythonCodeReviewAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> PythonCodeReviewObservation:
        observation, _, _, _ = self._step_transition(action, timeout_s=timeout_s, **kwargs)
        return observation

    def step_result(
        self,
        action: PythonCodeReviewAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[PythonCodeReviewObservation, float, bool, Dict[str, Any]]:
        """Gym-style helper used by local scripts and tests."""

        return self._step_transition(action, timeout_s=timeout_s, **kwargs)

    def _step_transition(
        self,
        action: PythonCodeReviewAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[PythonCodeReviewObservation, float, bool, Dict[str, Any]]:
        if self._state.done:
            reward = RewardDetails(
                value=_reward_value(0.05 + 0.25 * self._current_grade.score),
                reason="Episode already finished. Call reset() to continue.",
            )
            observation = self._build_observation(
                grade=self._current_grade,
                status="Episode already finished.",
                reward_details=reward,
            )
            return observation, reward.value, observation.done, {"task_id": observation.task_id, "score": observation.score}

        previous_grade = self._current_grade
        status = ""
        invalid_action = False
        code_changed = False
        use_hidden_grading = False
        action_error: str | None = None

        if action.action_type == "edit_code":
            if not action.code or not action.code.strip():
                invalid_action = True
                status = "edit_code requires a non-empty code payload."
                action_error = status
            else:
                code_changed = action.code != self._current_code
                self._current_code = action.code
                status = "Updated working copy from agent patch."
        elif action.action_type == "submit_solution":
            if action.code is not None and action.code.strip():
                code_changed = action.code != self._current_code
                self._current_code = action.code
            use_hidden_grading = True
            status = "Submission received for final grading."
        elif action.action_type == "run_tests":
            status = "Executed public validation suite."
        elif action.action_type == "analyze_code":
            status = "Generated static review summary."
        else:  # pragma: no cover
            invalid_action = True
            status = f"Unsupported action_type: {action.action_type}"
            action_error = status

        self._state.step_count += 1

        if invalid_action:
            current_grade = previous_grade
        else:
            current_grade, grade_error = self._safe_grade_task(
                self._task,
                self._current_code,
                include_hidden=use_hidden_grading,
                timeout_s=timeout_s or 3.0,
            )
            if grade_error:
                action_error = grade_error
                status = f"{status} Grading fallback used."
            if action.action_type == "analyze_code":
                status = self._analysis_status(current_grade)
            elif action.action_type == "run_tests":
                status = self._run_tests_status(current_grade, use_hidden_grading)
            elif action.action_type == "submit_solution":
                status = self._submission_status(current_grade)

        done = use_hidden_grading or self._state.step_count >= self._task.max_steps
        if self._state.step_count >= self._task.max_steps and not use_hidden_grading:
            status = f"{status} Step budget exhausted."

        reward_details = self._compute_reward(
            previous_grade=previous_grade,
            current_grade=current_grade,
            action=action,
            invalid_action=invalid_action,
            timed_out=current_grade.timed_out,
            code_changed=code_changed,
            final_submission=use_hidden_grading,
        )

        self._history.append(
            HistoryEntry(
                step=self._state.step_count,
                action_type=action.action_type,
                status=status,
                reward=reward_details.value,
            )
        )

        self._current_grade = current_grade
        self._last_reward = reward_details
        self._last_action_error = action_error
        attempts_remaining = max(self._task.max_steps - self._state.step_count, 0)

        self._state.task_id = self._task.task_id
        self._state.difficulty = self._task.difficulty
        self._state.task_kind = self._task.task_kind
        self._state.attempts_remaining = attempts_remaining
        self._state.current_code = self._current_code
        self._state.errors = self._format_errors(current_grade)
        self._state.test_results = self._format_test_results(current_grade)
        self._state.history = list(self._history)
        self._state.score = current_grade.score
        self._state.done = done

        observation = self._build_observation(
            grade=current_grade,
            status=status,
            reward_details=reward_details,
        )
        return observation, reward_details.value, observation.done, {
            "task_id": observation.task_id,
            "score": observation.score,
            "done": observation.done,
            "attempts_remaining": observation.attempts_remaining,
            "last_action_status": observation.last_action_status,
            "last_action_error": observation.last_action_error,
        }

    @property
    def state(self) -> PythonCodeReviewState:
        return self._state

    def _build_observation(
        self,
        *,
        grade: TaskGrade,
        status: str,
        reward_details: RewardDetails,
    ) -> PythonCodeReviewObservation:
        return PythonCodeReviewObservation(
            task_id=self._task.task_id,
            title=self._task.title,
            difficulty=self._task.difficulty,
            task_kind=self._task.task_kind,
            task_description=self._task.task_description,
            current_code=self._current_code,
            errors=self._format_errors(grade),
            test_results=self._format_test_results(grade),
            visible_tests=list(self._task.visible_tests),
            history=list(self._history),
            attempts_remaining=self._state.attempts_remaining,
            last_action_status=status,
            last_action_error=self._last_action_error,
            score=grade.score,
            reward=reward_details.value,
            done=self._state.done,
            reward_details=reward_details,
            metadata={
                "benchmark": "python_code_review_env",
                "goal": self._task.goal,
                "repo_summary": self._task.repo_summary,
                "changed_files": self._task.changed_files,
                "available_files": self._task.available_files,
                "grade_details": grade.details,
            },
        )

    def _compute_reward(
        self,
        *,
        previous_grade: TaskGrade,
        current_grade: TaskGrade,
        action: PythonCodeReviewAction,
        invalid_action: bool,
        timed_out: bool,
        code_changed: bool,
        final_submission: bool,
    ) -> RewardDetails:
        prev_score = previous_grade.score
        curr_score = current_grade.score
        prev_rate = safe_ratio(previous_grade.tests_passed, previous_grade.tests_total)
        curr_rate = safe_ratio(current_grade.tests_passed, current_grade.tests_total)
        prev_runtime = previous_grade.runtime_score
        curr_runtime = current_grade.runtime_score
        prev_compile_error = bool(str(previous_grade.details.get("compile_error", "")).strip())
        curr_compile_error = bool(str(current_grade.details.get("compile_error", "")).strip())

        syntax_reward = 0.14 if previous_grade.syntax_score < 0.9 and current_grade.syntax_score >= 0.9 else 0.0
        test_reward = round(max(curr_rate - prev_rate, 0.0) * 0.28, 3)
        progress_delta = round(max(curr_score - prev_score, 0.0) * 0.3, 3)
        quality_bonus = round(max(current_grade.quality_score - previous_grade.quality_score, 0.0) * 0.12, 3)
        runtime_bonus = round(max(curr_runtime - prev_runtime, 0.0) * 0.08, 3)
        error_reduction_bonus = 0.1 if prev_compile_error and not curr_compile_error else 0.0
        completion_bonus = 0.14 if final_submission and curr_rate >= 0.999 and curr_score >= 0.94 else 0.0
        correctness_bonus = 0.12 if final_submission and curr_score >= 0.94 and prev_score < 0.94 else 0.0

        invalid_action_penalty = round((0.04 + (0.08 * (1.0 - prev_score))) if invalid_action else 0.0, 3)
        timeout_penalty = round((0.06 + (0.08 * max(curr_runtime, prev_runtime))) if timed_out else 0.0, 3)
        regression_penalty = round(max(prev_score - curr_score, 0.0) * 0.25, 3)
        stagnation_penalty = round((0.02 + (0.05 * prev_score)) if action.action_type == "edit_code" and not code_changed else 0.0, 3)

        raw_value = (
            0.32 * curr_score
            + syntax_reward
            + test_reward
            + progress_delta
            + quality_bonus
            + error_reduction_bonus
            + completion_bonus
            + runtime_bonus
            + correctness_bonus
            - invalid_action_penalty
            - timeout_penalty
            - regression_penalty
            - stagnation_penalty
        )
        value = _reward_value(raw_value)

        reason_parts = []
        if syntax_reward:
            reason_parts.append("syntax fixed")
        if test_reward:
            reason_parts.append("public test progress")
        if progress_delta:
            reason_parts.append("overall score improved")
        if quality_bonus:
            reason_parts.append("code quality improved")
        if error_reduction_bonus:
            reason_parts.append("errors removed")
        if completion_bonus:
            reason_parts.append("task completed")
        if runtime_bonus:
            reason_parts.append("runtime improved")
        if correctness_bonus:
            reason_parts.append("full correctness bonus")
        if invalid_action_penalty:
            reason_parts.append("invalid action penalty")
        if timeout_penalty:
            reason_parts.append("timeout penalty")
        if regression_penalty:
            reason_parts.append("regression penalty")
        if stagnation_penalty:
            reason_parts.append("unchanged patch penalty")
        if not reason_parts:
            reason_parts.append("no meaningful state change")

        return RewardDetails(
            value=value,
            syntax_reward=syntax_reward,
            test_reward=test_reward,
            correctness_bonus=correctness_bonus,
            quality_bonus=quality_bonus,
            error_reduction_bonus=error_reduction_bonus,
            completion_bonus=completion_bonus,
            runtime_bonus=runtime_bonus,
            progress_delta=progress_delta,
            invalid_action_penalty=invalid_action_penalty,
            timeout_penalty=timeout_penalty,
            regression_penalty=regression_penalty,
            stagnation_penalty=stagnation_penalty,
            reason=", ".join(reason_parts),
            prev_score=prev_score,
            curr_score=curr_score,
            code_changed=code_changed,
        )

    def _format_errors(self, grade: TaskGrade) -> str:
        compile_error = str(grade.details.get("compile_error", "")).strip()
        if compile_error:
            return compile_error
        return "Code parses successfully."

    def _safe_grade_task(
        self,
        task: ReviewTask,
        code: str,
        *,
        include_hidden: bool,
        timeout_s: float = 3.0,
    ) -> tuple[TaskGrade, str | None]:
        try:
            return (
                grade_task(task, code, include_hidden=include_hidden, timeout_s=timeout_s),
                None,
            )
        except Exception as exc:  # pragma: no cover
            return _empty_grade(), f"{type(exc).__name__}: {exc}"

    def _format_test_results(self, grade: TaskGrade) -> str:
        parts = [grade.details.get("test_summary", "No test feedback available.")]
        benchmark = grade.details.get("benchmark")
        if isinstance(benchmark, dict):
            parts.append(
                "Benchmark: "
                f"candidate {benchmark['candidate_seconds']}s vs baseline {benchmark['baseline_seconds']}s "
                f"(x{benchmark['improvement_ratio']})."
            )
        elif isinstance(benchmark, str) and benchmark:
            parts.append(f"Benchmark: {benchmark}")
        return "\n".join(part for part in parts if part)

    def _analysis_status(self, grade: TaskGrade) -> str:
        notes = grade.details.get("quality_notes", [])
        quality_note = notes[0] if notes else "No major static quality issues detected."
        return (
            f"Syntax score {grade.syntax_score:.2f}; "
            f"public tests {grade.tests_passed}/{grade.tests_total}; "
            f"quality {grade.quality_score:.2f}. {quality_note}"
        )

    def _run_tests_status(self, grade: TaskGrade, include_hidden: bool) -> str:
        visibility = "full" if include_hidden else "public"
        return f"Ran {visibility} tests: {grade.tests_passed}/{grade.tests_total} passed."

    def _submission_status(self, grade: TaskGrade) -> str:
        runtime_text = ""
        if isinstance(grade.details.get("benchmark"), dict):
            runtime_text = f" runtime {grade.runtime_score:.2f};"
        return (
            f"Submission graded with score {grade.score:.2f}; "
            f"tests {grade.tests_passed}/{grade.tests_total};"
            f"{runtime_text} quality {grade.quality_score:.2f}."
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="python_code_review_env",
            description="Production-style Python code review environment with deterministic grading.",
            version="1.0.0",
        )
