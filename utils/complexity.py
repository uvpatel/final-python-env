"""Complexity heuristics for DSA-style and general Python code."""

from __future__ import annotations

from typing import Any, Dict


def estimate_complexity(parsed: Dict[str, Any], code: str) -> Dict[str, Any]:
    """Estimate cyclomatic complexity and rough Big-O heuristics."""

    cyclomatic = 1 + int(parsed.get("branch_count", 0))
    loop_depth = int(parsed.get("max_loop_depth", 0))
    uses_recursion = bool(parsed.get("uses_recursion", False))

    if loop_depth >= 3:
        time_complexity = "O(n^3)"
    elif loop_depth == 2:
        time_complexity = "O(n^2)"
    elif "sorted(" in code or ".sort(" in code:
        time_complexity = "O(n log n)"
    elif loop_depth == 1 or uses_recursion:
        time_complexity = "O(n)"
    else:
        time_complexity = "O(1)"

    if "append(" in code or "list(" in code or "dict(" in code or "set(" in code:
        space_complexity = "O(n)"
    else:
        space_complexity = "O(1)"

    complexity_penalty = min(0.99, 0.08 + (cyclomatic * 0.04) + (loop_depth * 0.12))
    return {
        "cyclomatic_complexity": cyclomatic,
        "time_complexity": time_complexity,
        "space_complexity": space_complexity,
        "complexity_penalty": round(complexity_penalty, 4),
    }
