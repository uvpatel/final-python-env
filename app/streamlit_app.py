"""Streamlit frontend for the multi-domain analyzer platform."""

from __future__ import annotations

import streamlit as st

from app.examples import EXAMPLES
from schemas.request import AnalyzeCodeRequest
from services.analysis_service import AnalysisService


analysis_service = AnalysisService()


def _analyze(code: str, context_window: str, traceback_text: str, domain_hint: str):
    """Run the analysis service with validated request payloads."""

    request = AnalyzeCodeRequest(
        code=code,
        context_window=context_window,
        traceback_text=traceback_text,
        domain_hint=domain_hint,  # type: ignore[arg-type]
    )
    return analysis_service.analyze(request)


def main() -> None:
    """Render the Streamlit UI."""

    st.set_page_config(page_title="Multi-Domain AI Code Analyzer", layout="wide")
    st.title("Multi-Domain AI Code Analyzer & Improvement System")
    st.caption("PyTorch-powered code review across DSA, Data Science, ML/DL, and Web backend code.")

    example_name = st.selectbox("Example input", list(EXAMPLES.keys()))
    example = EXAMPLES[example_name]
    auto_analyze = st.toggle("Real-time scoring", value=True)

    left, right = st.columns([1.2, 1.0])
    with left:
        code = st.text_area("Code input", value=example["code"], height=420)
        context_window = st.text_area("Context window", value=example["context_window"], height=100)
        traceback_text = st.text_area("Optional traceback / runtime hint", value=example["traceback_text"], height=100)
        domain_hint = st.selectbox("Domain hint", ["auto", "dsa", "data_science", "ml_dl", "web"], index=["auto", "dsa", "data_science", "ml_dl", "web"].index(example["domain_hint"]))
        analyze_clicked = st.button("Analyze Code", type="primary")

    result = None
    if code and (analyze_clicked or auto_analyze):
        result = _analyze(code, context_window, traceback_text, domain_hint)

    with right:
        if result is None:
            st.info("Paste code or load an example to start analysis.")
        else:
            metric_cols = st.columns(4)
            metric_cols[0].metric("Detected domain", result.detected_domain)
            metric_cols[1].metric("ML score", f"{result.score_breakdown.ml_score:.0%}")
            metric_cols[2].metric("Domain score", f"{result.score_breakdown.domain_score:.0%}")
            metric_cols[3].metric("Reward", f"{result.score_breakdown.reward:.0%}")
            st.bar_chart(result.domain_confidences)
            st.caption(result.summary)

    if result is not None:
        overview_tab, suggestions_tab, domain_tab, static_tab = st.tabs(
            ["Overview", "Suggestions", "Domain Detail", "Static Analysis"]
        )

        with overview_tab:
            st.subheader("Improvement Plan")
            for step in result.improvement_plan:
                st.write(f"- {step}")
            st.subheader("Complexity")
            st.write(
                {
                    "time_complexity": result.static_analysis.time_complexity,
                    "space_complexity": result.static_analysis.space_complexity,
                    "cyclomatic_complexity": result.static_analysis.cyclomatic_complexity,
                }
            )

        with suggestions_tab:
            st.subheader("Suggestions")
            for suggestion in result.domain_analysis.suggestions:
                st.write(f"- {suggestion}")
            if result.domain_analysis.issues:
                st.subheader("Issues")
                for issue in result.domain_analysis.issues:
                    st.write(f"- [{issue.severity}] {issue.title}: {issue.description}")

        with domain_tab:
            st.subheader("Domain Highlights")
            st.json(result.domain_analysis.highlights)
            st.write(f"Domain score: {result.domain_analysis.domain_score:.0%}")

        with static_tab:
            st.subheader("Static Analysis")
            st.json(result.static_analysis.model_dump())


if __name__ == "__main__":
    main()
