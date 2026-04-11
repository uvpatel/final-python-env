"""FastAPI backend for the multi-domain AI code analyzer."""

from __future__ import annotations

from fastapi import FastAPI

from schemas.request import AnalyzeCodeRequest
from schemas.response import AnalyzeCodeResponse
from services.analysis_service import AnalysisService


app = FastAPI(title="Multi-Domain AI Code Analyzer", version="2.0.0")
analysis_service = AnalysisService()


@app.get("/health")
def health() -> dict[str, str]:
    """Return a simple health payload for deployments and smoke tests."""

    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeCodeResponse)
def analyze_code(payload: AnalyzeCodeRequest) -> AnalyzeCodeResponse:
    """Analyze code across supported domains and return structured results."""

    return analysis_service.analyze(payload)
