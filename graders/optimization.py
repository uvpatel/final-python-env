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
    details = {
        "compile_error": compile_error,
        "quality_notes": quality["quality_notes"],
        "style_score": quality["style_score"],
        "visibility": "full" if include_hidden else "public",
    }

    if not compiled:
        partial = round(min(0.15, similarity_score(code, task.reference_code) * 0.15), 3)
        details["test_results"] = []
        details["test_summary"] = "Code does not compile."
        return base_grade(
            score=partial,
            syntax_score=0.0,
            tests_passed=0,
            tests_total=len(task.public_cases) + (len(task.hidden_cases) if include_hidden else 0),
            quality_score=0.0,
            runtime_score=0.0,
            timed_out=False,
            details=details,
        )

    cases = task.public_cases + (task.hidden_cases if include_hidden else [])
    result = execute_cases(code, task.function_name, cases, timeout_s=timeout_s)
    if result.get("timed_out"):
        details["test_results"] = []
        details["test_summary"] = result["error"]
        return base_grade(
            score=0.0,
            syntax_score=1.0,
            tests_passed=0,
            tests_total=len(cases),
            quality_score=quality["score"],
            runtime_score=0.0,
            timed_out=True,
            details=details,
        )
    if "error" in result:
        details["test_results"] = []
        details["test_summary"] = result["error"]
        return base_grade(
            score=0.0,
            syntax_score=1.0,
            tests_passed=0,
            tests_total=len(cases),
            quality_score=quality["score"],
            runtime_score=0.0,
            timed_out=False,
            details=details,
        )

    data = result["data"]
    pass_rate = data["passed"] / max(data["total"], 1)
    runtime_score = 0.0
    benchmark_summary = "Benchmark deferred until hidden evaluation."
    timed_out = False

    if include_hidden and pass_rate == 1.0:
        benchmark = benchmark_candidate(task, code, timeout_s=timeout_s)
        runtime_score = benchmark["runtime_score"]
        timed_out = benchmark.get("timed_out", False)
        benchmark_summary = benchmark["details"]
        if timed_out:
            runtime_score = 0.0

    details["test_results"] = data["results"]
    details["test_summary"] = summarize_results("Test results", data["results"])
    details["benchmark"] = benchmark_summary

    if include_hidden:
        score = 0.5 * pass_rate + 0.3 * runtime_score + 0.2 * quality["score"]
    else:
        score = 0.7 * pass_rate + 0.3 * quality["score"]

    return base_grade(
        score=score,
        syntax_score=1.0,
        tests_passed=data["passed"],
        tests_total=data["total"],
        quality_score=quality["score"],
        runtime_score=runtime_score,
        timed_out=timed_out,
        details=details,
    )
