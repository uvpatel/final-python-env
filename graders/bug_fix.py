"""Bug-fix task grader."""

from __future__ import annotations

try:
    from ..models import TaskGrade
    from ..tasks.catalog import ReviewTask
except ImportError:
    from models import TaskGrade
    from tasks.catalog import ReviewTask

from .shared import (
    base_grade,
    compile_code,
    composite_grade_score,
    component_score,
    execute_cases,
    quality_metrics,
    similarity_score,
    summarize_results,
)


def grade_bug_fix_task(
    task: ReviewTask,
    code: str,
    *,
    include_hidden: bool,
    timeout_s: float = 2.0,
) -> TaskGrade:
    """Grade a bug-fix task against public or full test suites."""

    compiled, compile_error = compile_code(code)
    quality = quality_metrics(code, task.function_name)
    similarity = similarity_score(code, task.reference_code)
    details = {
        "compile_error": compile_error,
        "quality_notes": quality["quality_notes"],
        "style_score": quality["style_score"],
        "visibility": "full" if include_hidden else "public",
    }

    if not compiled:
        details["test_results"] = []
        details["test_summary"] = "Code does not compile."
        return base_grade(
            score=composite_grade_score(
                correctness=0.0,
                quality=0.05,
                runtime=0.05,
                syntax=0.0,
                similarity=similarity,
                baseline=0.04,
                penalty=0.05,
            ),
            syntax_score=component_score(0.01),
            tests_passed=0,
            tests_total=len(task.public_cases) + (len(task.hidden_cases) if include_hidden else 0),
            quality_score=component_score(0.01),
            runtime_score=component_score(0.01),
            timed_out=False,
            details=details,
        )

    cases = task.public_cases + (task.hidden_cases if include_hidden else [])
    result = execute_cases(code, task.function_name, cases, timeout_s=timeout_s)
    if result.get("timed_out"):
        details["test_results"] = []
        details["test_summary"] = result["error"]
        return base_grade(
            score=composite_grade_score(
                correctness=0.10,
                quality=quality["score"],
                runtime=0.0,
                syntax=0.95,
                similarity=similarity,
                baseline=0.06,
                penalty=0.12,
            ),
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
        return base_grade(
            score=composite_grade_score(
                correctness=0.12,
                quality=quality["score"],
                runtime=0.0,
                syntax=0.95,
                similarity=similarity,
                baseline=0.06,
                penalty=0.08,
            ),
            syntax_score=component_score(0.95),
            tests_passed=0,
            tests_total=len(cases),
            quality_score=quality["score"],
            runtime_score=component_score(0.01),
            timed_out=False,
            details=details,
        )

    data = result["data"]
    pass_rate = data["passed"] / max(data["total"], 1)
    details["test_results"] = data["results"]
    details["test_summary"] = summarize_results("Test results", data["results"])
    return base_grade(
        score=composite_grade_score(
            correctness=pass_rate,
            quality=quality["score"],
            runtime=0.05,
            syntax=0.95,
            similarity=similarity,
            baseline=0.08,
        ),
        syntax_score=component_score(0.95),
        tests_passed=data["passed"],
        tests_total=data["total"],
        quality_score=quality["score"],
        runtime_score=component_score(0.01),
        timed_out=False,
        details=details,
    )
