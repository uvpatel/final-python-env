"""Analyzer for DSA and competitive-programming style Python code."""

from __future__ import annotations

from typing import Any, Dict

from schemas.response import AnalysisIssue, DomainAnalysis


def analyze_dsa_code(code: str, parsed: Dict[str, Any], complexity: Dict[str, Any]) -> DomainAnalysis:
    """Inspect algorithmic code for brute-force patterns and efficiency risks."""

    issues = []
    suggestions = []
    score = 0.7

    if parsed.get("max_loop_depth", 0) >= 2:
        issues.append(
            AnalysisIssue(
                title="Nested loops suggest brute-force behavior",
                severity="medium",
                description="The implementation scans the input multiple times, which is often avoidable in DSA problems.",
            )
        )
        suggestions.append("Consider replacing nested scans with a hashmap, prefix table, or sorted search strategy.")
        score -= 0.15

    if parsed.get("uses_recursion"):
        suggestions.append("Verify recursion depth and add memoization or iterative conversion if the input size can grow.")
        score -= 0.05

    if "sorted(" in code or ".sort(" in code:
        suggestions.append("Sorting is acceptable here, but validate whether a direct O(n) pass can remove the sort.")

    if not suggestions:
        suggestions.append("Document the intended time complexity and add edge-case checks for empty input and duplicates.")

    return DomainAnalysis(
        domain="dsa",
        domain_score=max(0.05, round(score, 4)),
        issues=issues,
        suggestions=suggestions,
        highlights={
            "time_complexity": complexity["time_complexity"],
            "space_complexity": complexity["space_complexity"],
            "max_loop_depth": float(parsed.get("max_loop_depth", 0)),
        },
    )
