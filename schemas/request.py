"""Request schemas for code analysis endpoints and UI."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


DomainHint = Literal["auto", "dsa", "data_science", "ml_dl", "web"]


class AnalyzeCodeRequest(BaseModel):
    """Validated input payload for multi-domain code analysis."""

    code: str = Field(..., min_length=1, description="Source code to analyze.")
    context_window: str = Field(default="", max_length=2000, description="Optional repository or task context.")
    traceback_text: str = Field(default="", max_length=2000, description="Optional runtime or test failure output.")
    domain_hint: DomainHint = Field(default="auto", description="Optional domain override when auto detection is not desired.")
