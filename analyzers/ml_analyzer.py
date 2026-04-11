"""Analyzer for machine-learning and deep-learning code."""

from __future__ import annotations

from typing import Any, Dict

from schemas.response import AnalysisIssue, DomainAnalysis


def analyze_ml_code(code: str, parsed: Dict[str, Any], complexity: Dict[str, Any]) -> DomainAnalysis:
    """Inspect training and inference logic for common ML / DL mistakes."""

    issues = []
    suggestions = []
    score = 0.74

    if "torch" in code and "model.eval()" not in code and "predict" in code.lower():
        issues.append(
            AnalysisIssue(
                title="Inference path may be missing eval mode",
                severity="high",
                description="Inference code should place the model in eval mode before prediction.",
            )
        )
        suggestions.append("Call model.eval() before inference to disable training-time behavior such as dropout.")
        score -= 0.18

    if "torch" in code and "no_grad" not in code and "predict" in code.lower():
        suggestions.append("Wrap inference in torch.no_grad() to reduce memory usage and avoid unnecessary gradient tracking.")
        score -= 0.12

    if parsed.get("calls_backward") and not parsed.get("calls_optimizer_step"):
        issues.append(
            AnalysisIssue(
                title="Backward pass without optimizer step",
                severity="medium",
                description="Gradients are computed, but the optimizer step is not obvious in the snippet.",
            )
        )
        suggestions.append("Ensure optimizer.step() and optimizer.zero_grad() are placed correctly in the training loop.")
        score -= 0.12

    if "CrossEntropyLoss" in code and "softmax(" in code:
        suggestions.append("CrossEntropyLoss expects raw logits; remove the explicit softmax before the loss when possible.")
        score -= 0.05

    if not suggestions:
        suggestions.append("Add explicit train/eval mode transitions and log validation metrics during training.")

    return DomainAnalysis(
        domain="ml_dl",
        domain_score=max(0.05, round(score, 4)),
        issues=issues,
        suggestions=suggestions,
        highlights={
            "uses_torch": float(parsed.get("uses_torch", False)),
            "has_eval_mode": float("model.eval()" in code),
            "has_no_grad": float("no_grad" in code),
            "time_complexity": complexity["time_complexity"],
        },
    )
