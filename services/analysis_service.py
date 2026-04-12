"""Orchestration layer for multi-domain code analysis."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

from analyzers import analyze_data_science_code, analyze_dsa_code, analyze_ml_code, analyze_web_code
from models import PyTorchCodeAnalyzerModel
from schemas.request import AnalyzeCodeRequest
from schemas.response import AnalyzeCodeResponse, DomainAnalysis, StaticAnalysisSummary
from services.reward_service import RewardService
from services.suggestion_service import SuggestionService
from utils import estimate_complexity, parse_code_structure


def _lint_score(parsed: Dict[str, Any]) -> float:
    """Convert structural smells into a normalized lint-style score."""

    score = 1.0
    if not parsed.get("syntax_valid", True):
        score -= 0.45
    score -= min(parsed.get("long_lines", 0), 5) * 0.03
    if parsed.get("tabs_used"):
        score -= 0.1
    if parsed.get("trailing_whitespace_lines"):
        score -= 0.05
    if parsed.get("docstring_ratio", 0.0) == 0.0 and parsed.get("function_names"):
        score -= 0.08
    return round(max(0.0, min(1.0, score)), 4)


class AnalysisService:
    """End-to-end analysis pipeline shared by API and UI."""

    def __init__(self) -> None:
        self._model: PyTorchCodeAnalyzerModel | None = None
        self.reward_service = RewardService()
        self.suggestion_service = SuggestionService()
        self._analyzers: Dict[str, Callable[[str, Dict[str, Any], Dict[str, Any]], DomainAnalysis]] = {
            "dsa": analyze_dsa_code,
            "data_science": analyze_data_science_code,
            "ml_dl": analyze_ml_code,
            "web": analyze_web_code,
        }

    @property
    def model(self) -> PyTorchCodeAnalyzerModel:
        if self._model is None:
            self._model = PyTorchCodeAnalyzerModel()
        return self._model

    def _heuristic_domain_scores(self, parsed: Dict[str, Any], code: str) -> Dict[str, float]:
        """Derive domain priors from imports and syntax-level hints."""

        scores = {
            "dsa": 0.2 + (0.15 if parsed.get("uses_recursion") else 0.0) + (0.15 if parsed.get("max_loop_depth", 0) >= 1 else 0.0),
            "data_science": 0.2 + (0.35 if parsed.get("uses_pandas") or parsed.get("uses_numpy") else 0.0),
            "ml_dl": 0.2 + (0.35 if parsed.get("uses_torch") or parsed.get("uses_sklearn") else 0.0),
            "web": 0.2 + (0.35 if parsed.get("uses_fastapi") or parsed.get("uses_flask") else 0.0) + (0.1 if parsed.get("route_decorators") else 0.0),
            "general": 0.2,
        }
        if "fastapi" in code.lower():
            scores["web"] += 0.1
        if "pandas" in code.lower() or "numpy" in code.lower():
            scores["data_science"] += 0.1
        if "torch" in code.lower():
            scores["ml_dl"] += 0.1
        if "while" in code or "for" in code:
            scores["dsa"] += 0.05
        return {key: round(min(value, 0.99), 4) for key, value in scores.items()}

    def analyze(self, request: AnalyzeCodeRequest) -> AnalyzeCodeResponse:
        """Run the complete multi-domain analysis pipeline."""

        started = time.perf_counter()
        parsed = parse_code_structure(request.code)
        complexity = estimate_complexity(parsed, request.code)
        model_prediction = self.model.predict(request.code, request.context_window, parsed)
        heuristic_scores = self._heuristic_domain_scores(parsed, request.code)

        combined_scores = {}
        for domain, heuristic_score in heuristic_scores.items():
            model_score = float(model_prediction["domain_scores"].get(domain, 0.2))
            combined_scores[domain] = round((0.6 * model_score) + (0.4 * heuristic_score), 4)

        detected_domain = request.domain_hint if request.domain_hint != "auto" else max(combined_scores, key=combined_scores.get)
        analyzer = self._analyzers.get(detected_domain)
        domain_analysis = (
            analyzer(request.code, parsed, complexity)
            if analyzer is not None
            else DomainAnalysis(
                domain="general",
                domain_score=0.6,
                issues=[],
                suggestions=["Add stronger domain-specific context for deeper analysis."],
                highlights={},
            )
        )

        lint_score = _lint_score(parsed)
        score_breakdown = self.reward_service.compute(
            ml_score=float(model_prediction["ml_quality_score"]),
            domain_score=domain_analysis.domain_score,
            lint_score=lint_score,
            complexity_penalty=float(complexity["complexity_penalty"]),
        )
        static_analysis = StaticAnalysisSummary(
            syntax_valid=bool(parsed["syntax_valid"]),
            syntax_error=str(parsed["syntax_error"]),
            cyclomatic_complexity=int(complexity["cyclomatic_complexity"]),
            line_count=int(parsed["line_count"]),
            max_loop_depth=int(parsed["max_loop_depth"]),
            time_complexity=str(complexity["time_complexity"]),
            space_complexity=str(complexity["space_complexity"]),
            detected_imports=list(parsed["imports"]),
            code_smells=list(parsed["code_smells"]),
        )
        improvement_plan = self.suggestion_service.build_improvement_plan(
            domain_analysis=domain_analysis,
            static_analysis=static_analysis,
        )
        summary = (
            f"Detected `{detected_domain}` code with a model score of {score_breakdown.ml_score:.0%}, "
            f"domain score {score_breakdown.domain_score:.0%}, and final reward {score_breakdown.reward:.0%}."
        )
        return AnalyzeCodeResponse(
            detected_domain=detected_domain,  # type: ignore[arg-type]
            domain_confidences=combined_scores,
            score_breakdown=score_breakdown,
            static_analysis=static_analysis,
            domain_analysis=domain_analysis,
            improvement_plan=improvement_plan,
            model_backend=str(model_prediction["backend_name"]),
            model_id=str(model_prediction["model_id"]),
            summary=summary,
            context_window=request.context_window,
            analysis_time_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )
