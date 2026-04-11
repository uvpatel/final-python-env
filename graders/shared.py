"""Shared deterministic grading helpers."""

from __future__ import annotations

import ast
import difflib
import math
import multiprocessing as mp
import time
import traceback
from typing import Any, Callable, Dict, List

try:
    from ..openenv_models import TaskGrade
    from ..tasks.catalog import CallCase, ReviewTask
except ImportError:
    from openenv_models import TaskGrade
    from tasks.catalog import CallCase, ReviewTask


STRICT_SCORE_MIN = 0.01
STRICT_SCORE_MAX = 0.99
POOR_SCORE = 0.1
NEAR_PERFECT_SCORE = 0.95


def finite_float(value: Any, fallback: float = STRICT_SCORE_MIN) -> float:
    """Convert a value into a finite float with a deterministic fallback."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(numeric) or math.isinf(numeric):
        return fallback
    return numeric


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a floating-point value to a closed interval."""

    numeric = finite_float(value, fallback=lower)
    return max(lower, min(upper, numeric))


def strict_score(value: Any, lower: float = STRICT_SCORE_MIN, upper: float = STRICT_SCORE_MAX) -> float:
    """Clamp a score to the OpenEnv-safe open interval (0, 1)."""

    score = max(lower, min(upper, finite_float(value, fallback=lower)))
    score = round(score, 3)
    assert 0 < score < 1, f"Invalid score: {score}"
    return score


def shaped_score(progress: Any, floor: float = POOR_SCORE, ceiling: float = NEAR_PERFECT_SCORE) -> float:
    """Map progress in [0, 1] to a shaped score band within (0, 1)."""

    bounded_progress = clamp(finite_float(progress, fallback=0.0))
    score = floor + (ceiling - floor) * bounded_progress
    score = max(STRICT_SCORE_MIN, min(score, STRICT_SCORE_MAX))
    score = round(score, 3)
    assert 0 < score < 1, f"Invalid score: {score}"
    return score


def score_from_checks(passed: int, total: int, floor: float = POOR_SCORE, ceiling: float = NEAR_PERFECT_SCORE) -> float:
    """Convert discrete checks into a smoothly shaped score."""

    return shaped_score(safe_ratio(passed, total), floor=floor, ceiling=ceiling)


def safe_ratio(numerator: Any, denominator: Any) -> float:
    """Return a stable ratio in [0, 1] that never raises or produces NaN."""

    denom = int(finite_float(denominator, fallback=0.0))
    if denom <= 0:
        return 0.0
    numer = finite_float(numerator, fallback=0.0)
    return clamp(numer / denom)


def component_score(value: Any) -> float:
    """Normalize component scores such as syntax, quality, and runtime."""

    return strict_score(value)


def compile_code(code: str) -> tuple[bool, str]:
    """Return whether code compiles and the syntax error, if any."""

    try:
        compile(code, "<candidate>", "exec")
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} (line {exc.lineno}, column {exc.offset})"
    except Exception as exc:  # pragma: no cover
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def similarity_score(candidate: str, reference: str) -> float:
    """Compute a stable text similarity score in [0, 1]."""

    return difflib.SequenceMatcher(a=candidate.strip(), b=reference.strip()).ratio()


def _queue_worker(
    worker: Callable[[Dict[str, Any]], Dict[str, Any]],
    payload: Dict[str, Any],
    queue: Any,
) -> None:
    try:
        queue.put({"ok": True, "data": worker(payload)})
    except Exception as exc:  # pragma: no cover
        queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            }
        )


def run_with_timeout(
    worker: Callable[[Dict[str, Any]], Dict[str, Any]],
    payload: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    """Execute a worker in a subprocess and terminate on timeout."""

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_queue_worker, args=(worker, payload, queue))
    process.start()
    process.join(timeout_s)

    if process.is_alive():
        process.terminate()
        process.join()
        return {"timed_out": True, "error": f"Execution exceeded {timeout_s:.1f}s timeout."}

    if queue.empty():
        return {"timed_out": False, "error": "Worker exited without returning a result."}

    message = queue.get()
    if not message["ok"]:
        return {
            "timed_out": False,
            "error": f"{message['error']}\n{message['traceback']}",
        }
    return {"timed_out": False, "data": message["data"]}


def _execute_cases_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    namespace: Dict[str, Any] = {}
    exec(payload["code"], namespace)
    func = namespace[payload["function_name"]]
    results: List[Dict[str, Any]] = []

    for case in payload["cases"]:
        try:
            actual = func(*case["args"], **case["kwargs"])
            passed = actual == case["expected"]
            actual_repr = repr(actual)
        except Exception as exc:
            passed = False
            actual_repr = f"{type(exc).__name__}: {exc}"

        results.append(
            {
                "label": case["label"],
                "passed": passed,
                "expected": repr(case["expected"]),
                "actual": actual_repr,
            }
        )

    passed_total = sum(1 for item in results if item["passed"])
    return {"passed": passed_total, "total": len(results), "results": results}


def execute_cases(code: str, function_name: str, cases: List[CallCase], timeout_s: float) -> Dict[str, Any]:
    """Run function test cases in a subprocess."""

    payload = {
        "code": code,
        "function_name": function_name,
        "cases": [
            {"label": case.label, "args": case.args, "kwargs": case.kwargs, "expected": case.expected}
            for case in cases
        ],
    }
    return run_with_timeout(_execute_cases_worker, payload, timeout_s=timeout_s)


class _LoopDepthVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.depth = 0
        self.max_depth = 0

    def _visit_loop(self, node: ast.AST) -> None:
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        self.generic_visit(node)
        self.depth -= 1

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self._visit_loop(node)

    def visit_While(self, node: ast.While) -> None:  # noqa: N802
        self._visit_loop(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:  # noqa: N802
        self._visit_loop(node)


def quality_metrics(code: str, function_name: str) -> Dict[str, Any]:
    """Compute deterministic AST/style quality metrics."""

    compiled, error = compile_code(code)
    if not compiled:
        return {
            "score": component_score(STRICT_SCORE_MIN),
            "style_score": component_score(STRICT_SCORE_MIN),
            "quality_notes": [error],
            "max_loop_depth": 99,
        }

    tree = ast.parse(code)
    function_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        ),
        None,
    )

    notes: List[str] = []
    score = 0.0

    if function_node is not None:
        score += 0.2
    else:
        notes.append(f"Expected function {function_name!r} is missing.")

    lines = [line.rstrip("\n") for line in code.splitlines()]
    long_lines = [index + 1 for index, line in enumerate(lines) if len(line) > 88]
    trailing_whitespace = [index + 1 for index, line in enumerate(lines) if line.rstrip() != line]
    uses_tabs = any("\t" in line for line in lines)

    style_score = 0.0
    if not long_lines:
        score += 0.15
        style_score += 0.5
    else:
        notes.append(f"Lines longer than 88 characters: {long_lines[:3]}")

    if not trailing_whitespace and not uses_tabs:
        score += 0.15
        style_score += 0.5
    else:
        notes.append("Remove tabs or trailing whitespace for cleaner style.")

    if function_node is not None:
        if ast.get_docstring(function_node):
            score += 0.1
        else:
            notes.append("Add a short docstring to explain the function contract.")

        visitor = _LoopDepthVisitor()
        visitor.visit(function_node)
        if visitor.max_depth <= 1:
            score += 0.15
        elif visitor.max_depth == 2:
            score += 0.08
            notes.append("Loop nesting is still higher than necessary.")
        else:
            notes.append("Refactor nested loops to improve readability and runtime.")

        names = [node.id for node in ast.walk(function_node) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)]
        meaningful_names = [name for name in names if len(name) >= 3]
        if names:
            score += 0.1 * (len(meaningful_names) / len(names))

        function_length = (function_node.end_lineno or function_node.lineno) - function_node.lineno + 1
        if function_length <= 25:
            score += 0.1
        elif function_length <= 40:
            score += 0.05
            notes.append("The function can be shortened or decomposed further.")
        else:
            notes.append("The function is long enough to justify refactoring.")

        max_loop_depth = visitor.max_depth
    else:
        max_loop_depth = 0

    source_hints = ("Counter(", "defaultdict(", "set(", "dict(", "sorted(", "sum(", " any(", " all(", " for ")
    if any(hint in code for hint in source_hints):
        score += 0.15

    return {
        "score": component_score(clamp(score)),
        "style_score": component_score(clamp(style_score)),
        "quality_notes": notes,
        "max_loop_depth": max_loop_depth,
    }


def build_benchmark_events(config: Dict[str, int]) -> List[Dict[str, Any]]:
    """Generate deterministic benchmark data without randomness."""

    user_pool = config["user_pool"]
    events_per_user = config["events_per_user"]
    events: List[Dict[str, Any]] = []

    for user_index in range(user_pool):
        user_id = f"user-{user_index:03d}"
        for event_index in range(events_per_user):
            status = "active" if (user_index + event_index) % 3 != 0 else "inactive"
            events.append({"user_id": user_id, "status": status, "minute": event_index})
            if event_index % 6 == 0:
                events.append({"user_id": user_id, "status": status, "minute": event_index})

    return events


def _benchmark_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    candidate_ns: Dict[str, Any] = {}
    baseline_ns: Dict[str, Any] = {}
    exec(payload["candidate_code"], candidate_ns)
    exec(payload["baseline_code"], baseline_ns)

    candidate = candidate_ns[payload["function_name"]]
    baseline = baseline_ns[payload["function_name"]]
    benchmark_events = payload["events"]
    iterations = payload["iterations"]

    baseline_output = baseline(benchmark_events)
    candidate_output = candidate(benchmark_events)
    if candidate_output != baseline_output:
        raise AssertionError("Candidate output diverges from baseline on benchmark data.")

    def _timed(fn: Callable[[Any], Any]) -> float:
        start = time.perf_counter()
        for _ in range(iterations):
            fn(benchmark_events)
        return time.perf_counter() - start

    baseline_seconds = _timed(baseline)
    candidate_seconds = _timed(candidate)
    return {"baseline_seconds": baseline_seconds, "candidate_seconds": candidate_seconds}


def benchmark_candidate(task: ReviewTask, code: str, timeout_s: float) -> Dict[str, Any]:
    """Benchmark a candidate solution against the starter implementation."""

    if not task.benchmark_config:
        return {"runtime_score": component_score(STRICT_SCORE_MIN), "details": "No benchmark configured."}

    events = build_benchmark_events(task.benchmark_config)
    payload = {
        "candidate_code": code,
        "baseline_code": task.starter_code,
        "function_name": task.function_name,
        "events": events,
        "iterations": task.benchmark_config.get("iterations", 5),
    }
    result = run_with_timeout(_benchmark_worker, payload, timeout_s=timeout_s)
    if result.get("timed_out"):
        return {"runtime_score": component_score(STRICT_SCORE_MIN), "timed_out": True, "details": result["error"]}
    if "error" in result:
        return {"runtime_score": component_score(STRICT_SCORE_MIN), "timed_out": False, "details": result["error"]}

    data = result["data"]
    baseline_seconds = float(data["baseline_seconds"])
    candidate_seconds = float(data["candidate_seconds"])
    improvement_ratio = baseline_seconds / max(candidate_seconds, 1e-9)
    runtime_score = component_score(clamp((improvement_ratio - 1.0) / 1.5))
    return {
        "runtime_score": runtime_score,
        "timed_out": False,
        "details": {
            "baseline_seconds": round(baseline_seconds, 6),
            "candidate_seconds": round(candidate_seconds, 6),
            "improvement_ratio": round(improvement_ratio, 3),
        },
    }


def summarize_results(prefix: str, results: List[Dict[str, Any]]) -> str:
    """Render concise test output."""

    if not results:
        return f"{prefix}: no tests were executed."

    lines = [prefix]
    for item in results:
        marker = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- {marker} {item['label']}: expected {item['expected']}, got {item['actual']}")
    return "\n".join(lines)


def base_grade(
    *,
    score: float,
    syntax_score: float,
    tests_passed: int,
    tests_total: int,
    quality_score: float,
    runtime_score: float,
    timed_out: bool,
    details: Dict[str, Any],
) -> TaskGrade:
    """Create a normalized TaskGrade payload."""

    safe_score = strict_score(score)
    safe_syntax_score = component_score(syntax_score)
    safe_quality_score = component_score(quality_score)
    safe_runtime_score = component_score(runtime_score)

    return TaskGrade(
        score=safe_score,
        syntax_score=safe_syntax_score,
        tests_passed=tests_passed,
        tests_total=tests_total,
        quality_score=safe_quality_score,
        runtime_score=safe_runtime_score,
        timed_out=timed_out,
        details=details,
    )
