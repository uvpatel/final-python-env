"""Analyzer for data-science oriented Python code."""

from __future__ import annotations

from typing import Any, Dict

from schemas.response import AnalysisIssue, DomainAnalysis


def analyze_data_science_code(code: str, parsed: Dict[str, Any], complexity: Dict[str, Any]) -> DomainAnalysis:
    """Inspect pandas and numpy code for vectorization and leakage concerns."""

    issues = []
    suggestions = []
    score = 0.72

    if "iterrows(" in code or "itertuples(" in code:
        issues.append(
            AnalysisIssue(
                title="Row-wise dataframe iteration detected",
                severity="medium",
                description="Looping through dataframe rows is usually slower and less scalable than vectorized operations.",
            )
        )
        suggestions.append("Use vectorized pandas or numpy expressions instead of row-wise iteration.")
        score -= 0.18

    if "inplace=True" in code:
        suggestions.append("Avoid inplace mutation to keep data pipelines easier to reason about and test.")
        score -= 0.05

    if "fit_transform(" in code and "train_test_split" not in code:
        issues.append(
            AnalysisIssue(
                title="Potential data leakage risk",
                severity="high",
                description="Feature transforms appear before an explicit train/test split.",
            )
        )
        suggestions.append("Split train and validation data before fitting stateful preprocessing steps.")
        score -= 0.2

    if not suggestions:
        suggestions.append("Add schema assumptions and null-handling checks for production data quality.")

    return DomainAnalysis(
        domain="data_science",
        domain_score=max(0.05, round(score, 4)),
        issues=issues,
        suggestions=suggestions,
        highlights={
            "vectorization_risk": float("iterrows(" in code or "itertuples(" in code),
            "time_complexity": complexity["time_complexity"],
            "uses_pandas": float(parsed.get("uses_pandas", False)),
        },
    )
