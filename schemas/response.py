"""Response schemas for the multi-domain analysis platform."""

from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


DomainType = Literal["dsa", "data_science", "ml_dl", "web", "general"]
Severity = Literal["low", "medium", "high"]


class AnalysisIssue(BaseModel):
    """One detected issue or risk in the code snippet."""

    title: str
    severity: Severity
    description: str
    line_hint: int | None = None


class StaticAnalysisSummary(BaseModel):
    """Language-agnostic static-analysis signals."""

    syntax_valid: bool
    syntax_error: str = ""
    cyclomatic_complexity: int = Field(..., ge=1)
    line_count: int = Field(..., ge=0)
    max_loop_depth: int = Field(..., ge=0)
    time_complexity: str = "Unknown"
    space_complexity: str = "Unknown"
    detected_imports: List[str] = Field(default_factory=list)
    code_smells: List[str] = Field(default_factory=list)


class DomainAnalysis(BaseModel):
    """Domain-specific analysis payload returned by an analyzer."""

    domain: DomainType
    domain_score: float = Field(..., ge=0.0, le=1.0)
    issues: List[AnalysisIssue] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    highlights: Dict[str, float | str] = Field(default_factory=dict)


class ScoreBreakdown(BaseModel):
    """Reward inputs and final normalized score."""

    ml_score: float = Field(..., ge=0.0, le=1.0)
    domain_score: float = Field(..., ge=0.0, le=1.0)
    lint_score: float = Field(..., ge=0.0, le=1.0)
    complexity_penalty: float = Field(..., ge=0.0, le=1.0)
    quality_signal: float = Field(..., ge=0.0, le=1.0)
    error_reduction_signal: float = Field(..., ge=0.0, le=1.0)
    completion_signal: float = Field(..., ge=0.0, le=1.0)
    reward: float = Field(..., ge=0.0, le=1.0)


class AnalyzeCodeResponse(BaseModel):
    """Top-level structured output for API and UI consumers."""

    detected_domain: DomainType
    domain_confidences: Dict[str, float]
    score_breakdown: ScoreBreakdown
    static_analysis: StaticAnalysisSummary
    domain_analysis: DomainAnalysis
    improvement_plan: List[str] = Field(default_factory=list)
    model_backend: str
    model_id: str
    summary: str
    context_window: str = ""
    analysis_time_ms: float = Field(..., ge=0.0)
