"""Typed models for TorchReview Copilot outputs and examples."""

from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


IssueLabel = Literal["syntax", "logic", "performance"]
RiskLevel = Literal["low", "medium", "high"]


class TriageSignal(BaseModel):
    """One extracted signal used during issue classification."""

    name: str
    value: str
    impact: Literal["syntax", "logic", "performance", "mixed"] = "mixed"
    weight: float = Field(..., ge=0.0, le=1.0)
    evidence: str = ""


class PrototypeMatch(BaseModel):
    """Nearest known bug pattern from the built-in task catalog."""

    task_id: str
    title: str
    label: IssueLabel
    similarity: float = Field(..., ge=0.0, le=1.0)
    summary: str
    rationale: str


class TriageExample(BaseModel):
    """Example payload exposed in the demo UI."""

    key: str
    title: str
    label: IssueLabel
    summary: str
    code: str
    traceback_text: str
    context_window: str
    task_id: str


class TriagePrototype(BaseModel):
    """Canonical issue-pattern representation embedded by the triage engine."""

    task_id: str
    title: str
    label: IssueLabel
    summary: str
    reference_text: str
    starter_code: str
    reference_code: str
    traceback_text: str


class TriageResult(BaseModel):
    """Structured output produced by the triage pipeline."""

    issue_label: IssueLabel
    confidence_scores: Dict[str, float]
    repair_risk: RiskLevel
    ml_quality_score: float = Field(..., ge=0.0, le=1.0)
    lint_score: float = Field(..., ge=0.0, le=1.0)
    complexity_penalty: float = Field(..., ge=0.0, le=1.0)
    reward_score: float = Field(..., ge=0.0, le=1.0)
    summary: str
    matched_pattern: PrototypeMatch
    repair_plan: List[str]
    suggested_next_action: str
    extracted_signals: List[TriageSignal] = Field(default_factory=list)
    model_backend: str
    model_id: str
    inference_notes: List[str] = Field(default_factory=list)
    analysis_time_ms: float = Field(..., ge=0.0)
