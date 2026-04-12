"""Analyzer for FastAPI and backend web-service code."""

from __future__ import annotations

from typing import Any, Dict

from schemas.response import AnalysisIssue, DomainAnalysis


def analyze_web_code(code: str, parsed: Dict[str, Any], complexity: Dict[str, Any]) -> DomainAnalysis:
    """Inspect API code for validation, routing, and backend safety concerns."""

    issues = []
    suggestions = []
    score = 0.76

    route_decorators = set(parsed.get("route_decorators", []))
    if route_decorators and not parsed.get("uses_pydantic"):
        issues.append(
            AnalysisIssue(
                title="Request validation model is missing",
                severity="high",
                description="Route handlers appear present, but no obvious Pydantic validation layer was detected.",
            )
        )
        suggestions.append("Add Pydantic request and response models for strict validation and type-safe contracts.")
        score -= 0.2

    if {"get", "post", "put", "delete"} & route_decorators and "async def" not in code:
        suggestions.append("Prefer async FastAPI endpoints when the route performs I/O or awaits downstream services.")
        score -= 0.08

    if "request.json()" in code or "request.body()" in code:
        suggestions.append("Validate raw request payloads before use; avoid trusting unchecked JSON input.")
        score -= 0.08

    if not suggestions:
        suggestions.append("Add domain-specific response models and centralize dependency injection for cleaner API structure.")

    return DomainAnalysis(
        domain="web",
        domain_score=max(0.05, min(0.99, round(score, 4))),
        issues=issues,
        suggestions=suggestions,
        highlights={
            "route_count": float(len(route_decorators)),
            "uses_validation": float(parsed.get("uses_pydantic", False)),
            "time_complexity": complexity["time_complexity"],
        },
    )
