"""Syntax task grader."""

from __future__ import annotations

try:
    from ..openenv_models import TaskGrade
    from ..tasks.catalog import ReviewTask
except ImportError:
    from openenv_models import TaskGrade
    from tasks.catalog import ReviewTask

from .shared import (
    base_grade,
    compile_code,
    component_score,
    execute_cases,
    quality_metrics,
    shaped_score,
    similarity_score,
    summarize_results,
)


def grade_syntax_task(task: ReviewTask, code: str, timeout_s: float = 2.0) -> TaskGrade:
    """Grade a syntax-fix task deterministically."""

    compiled, compile_error = compile_code(code)
    quality = quality_metrics(code, task.function_name)
    details = {
        "compile_error": compile_error,
        "quality_notes": quality["quality_notes"],
        "style_score": quality["style_score"],
    }

    if not compiled:
        progress = 0.05 + 0.2 * similarity_score(code, task.reference_code)
        details["test_results"] = []
        details["test_summary"] = "Code does not compile yet."
        return base_grade(
            score=shaped_score(progress),
            syntax_score=component_score(0.01),
            tests_passed=0,
            tests_total=len(task.public_cases) + len(task.hidden_cases),
            quality_score=component_score(0.01),
            runtime_score=component_score(0.01),
            timed_out=False,
            details=details,
        )

    cases = task.public_cases + task.hidden_cases
    result = execute_cases(code, task.function_name, cases, timeout_s=timeout_s)
    if result.get("timed_out"):
        details["test_results"] = []
        details["test_summary"] = result["error"]
        progress = 0.2 + 0.25 * quality["score"]
        return base_grade(
            score=shaped_score(progress),
            syntax_score=component_score(0.95),
            tests_passed=0,
            tests_total=len(cases),
            quality_score=quality["score"],
            runtime_score=component_score(0.01),
            timed_out=True,
            details=details,
        )
    if "error" in result:
        details["test_results"] = []
        details["test_summary"] = result["error"]
        progress = 0.18 + 0.2 * quality["score"]
        return base_grade(
            score=shaped_score(progress),
            syntax_score=component_score(0.95),
            tests_passed=0,
            tests_total=len(cases),
            quality_score=quality["score"],
            runtime_score=component_score(0.01),
            timed_out=False,
            details=details,
        )

    data = result["data"]
    details["test_results"] = data["results"]
    details["test_summary"] = summarize_results("Validation checks", data["results"])
    pass_rate = data["passed"] / max(data["total"], 1)
    progress = min(1.0, 0.15 + 0.75 * pass_rate + 0.1 * quality["score"])
    return base_grade(
        score=shaped_score(progress),
        syntax_score=component_score(0.95),
        tests_passed=data["passed"],
        tests_total=data["total"],
        quality_score=quality["score"],
        runtime_score=component_score(0.01),
        timed_out=False,
        details=details,
    )
