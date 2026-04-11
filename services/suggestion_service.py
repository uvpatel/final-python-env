"""Suggestion and improvement-plan generation for analyzed code."""

from __future__ import annotations

from schemas.response import DomainAnalysis, StaticAnalysisSummary


class SuggestionService:
    """Build high-signal improvement steps from analysis output."""

    def build_improvement_plan(self, *, domain_analysis: DomainAnalysis, static_analysis: StaticAnalysisSummary) -> list[str]:
        """Return a compact three-step plan optimized for developer action."""

        primary_issue = (
            domain_analysis.issues[0].description
            if domain_analysis.issues
            else "Stabilize correctness first and keep the public behavior explicit."
        )

        step_one = f"Step 1 - Correctness and safety: {primary_issue}"
        step_two = "Step 2 - Edge cases: test empty inputs, boundary values, malformed payloads, and failure-mode behavior explicitly."
        step_three = "Step 3 - Scalability: reduce repeated scans, lower cyclomatic complexity, and benchmark the path on realistic input sizes."

        if domain_analysis.suggestions:
            step_three = f"{step_three} Priority hint: {domain_analysis.suggestions[0]}"
        if not static_analysis.syntax_valid:
            step_one = f"Step 1 - Correctness and safety: fix the syntax error first ({static_analysis.syntax_error})."
        return [step_one, step_two, step_three]
