"""Optimization task grader."""

from __future__ import annotations

try:
    from ..models import TaskGrade
    from ..tasks.catalog import ReviewTask
except ImportError:
    from models import TaskGrade
    from tasks.catalog import ReviewTask

from .shared import (
    base_grade,
    benchmark_candidate,
    compile_code,
    composite_grade_score,
    component_score,
    execute_cases,
    quality_metrics,
    similarity_score,
    summarize_results,
)


def grade_optimization_task(
    task: ReviewTask,
    code: str,
    *,
    include_hidden: bool,
    timeout_s: float = 3.0,
) -> TaskGrade:
    """Grade an optimization/refactor task with correctness, quality, and runtime."""

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
                runtime=0.0,
                syntax=0.0,
                similarity=similarity,
                baseline=0.04,
                penalty=0.06,
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
                correctness=0.08,
                quality=quality["score"],
                runtime=0.0,
                syntax=0.95,
                similarity=similarity,
                baseline=0.05,
                penalty=0.14,
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
                correctness=0.10,
                quality=quality["score"],
                runtime=0.0,
                syntax=0.95,
                similarity=similarity,
                baseline=0.05,
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
    runtime_score = component_score(0.01)
    benchmark_summary = "Benchmark deferred until hidden evaluation."
    timed_out = False

    if include_hidden and pass_rate == 1.0:
        benchmark = benchmark_candidate(task, code, timeout_s=timeout_s)
        runtime_score = benchmark["runtime_score"]
        timed_out = benchmark.get("timed_out", False)
        benchmark_summary = benchmark["details"]
        if timed_out:
            runtime_score = component_score(0.01)

    details["test_results"] = data["results"]
    details["test_summary"] = summarize_results("Test results", data["results"])
    details["benchmark"] = benchmark_summary

    runtime_progress = 0.0 if benchmark_summary == "Benchmark deferred until hidden evaluation." else runtime_score
    return base_grade(
        score=composite_grade_score(
            correctness=pass_rate,
            quality=quality["score"],
            runtime=runtime_progress if include_hidden else 0.10,
            syntax=0.95,
            similarity=similarity,
            baseline=0.08 if include_hidden else 0.07,
            penalty=0.10 if timed_out else 0.0,
        ),
        syntax_score=component_score(0.95),
        tests_passed=data["passed"],
        tests_total=data["total"],
        quality_score=quality["score"],
        runtime_score=runtime_score,
        timed_out=timed_out,
        details=details,
    )
