"""Gradio UI for TorchReview Copilot."""

from __future__ import annotations

from html import escape

import gradio as gr

try:
    from ..triage import get_default_engine
except ImportError:
    from triage import get_default_engine


CSS = """
:root {
  --paper: #f6f1e8;
  --ink: #162521;
  --accent: #d95d39;
  --panel: #fffdf8;
  --border: #d6c4b8;
  --muted: #5f6f67;
  --good: #2d7d62;
  --warn: #b76516;
  --high: #b23a48;
}

body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(247, 197, 159, 0.35), transparent 35%),
    linear-gradient(135deg, #f9f6ef 0%, #efe5d3 100%);
  color: var(--ink);
  font-family: Georgia, "Times New Roman", serif;
}

.gradio-container {
  max-width: 1260px !important;
}

.hero-card,
.metric-card,
.subtle-card {
  background: rgba(255, 253, 248, 0.95);
  border: 1px solid var(--border);
  border-radius: 20px;
  box-shadow: 0 16px 40px rgba(22, 37, 33, 0.08);
}

.hero-card {
  padding: 28px 30px;
  margin-bottom: 12px;
}

.metric-card,
.subtle-card {
  padding: 20px 22px;
}

.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 12px;
  color: var(--accent);
  margin-bottom: 10px;
}

.hero-title {
  font-size: 44px;
  line-height: 1.05;
  margin: 0 0 10px;
}

.hero-copy {
  margin: 0;
  font-size: 18px;
  line-height: 1.55;
  color: var(--muted);
}

.summary-title {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  margin-bottom: 14px;
}

.pill {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  background: #efe5d3;
}

.pill.low { color: var(--good); }
.pill.medium { color: var(--warn); }
.pill.high { color: var(--high); }

.summary-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 16px;
}

.summary-stat {
  background: #fff7ef;
  border-radius: 14px;
  padding: 12px 14px;
  border: 1px solid rgba(214, 196, 184, 0.8);
}

.summary-stat strong {
  display: block;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin-bottom: 6px;
}

.radar-wrap {
  display: grid;
  gap: 12px;
}

.bar {
  display: grid;
  gap: 6px;
}

.bar-head {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  color: var(--muted);
}

.bar-track {
  width: 100%;
  height: 12px;
  background: #f2e5d6;
  border-radius: 999px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  border-radius: 999px;
}

.matched-box {
  background: #fff7ef;
  border: 1px solid rgba(214, 196, 184, 0.8);
  border-radius: 16px;
  padding: 14px;
}

.how-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.how-step {
  background: rgba(255, 253, 248, 0.9);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px;
}

@media (max-width: 900px) {
  .hero-title {
    font-size: 34px;
  }

  .summary-grid,
  .how-grid {
    grid-template-columns: 1fr;
  }
}
"""


def _default_outputs() -> tuple[str, str, str, str, str]:
    return (
        "<div class='metric-card'><div class='eyebrow'>Awaiting Analysis</div><p class='hero-copy'>Paste Python code, add an optional traceback, or load one of the built-in examples.</p></div>",
        "<div class='metric-card'><div class='eyebrow'>Live Triage Radar</div><p class='hero-copy'>Confidence bars will appear after the first analysis run.</p></div>",
        "### Improvement Plan\nAnalyze a sample to generate syntax, edge-case, and scalability recommendations.",
        "### Known Pattern Match\nThe nearest OpenEnv task will be highlighted here after inference runs.",
        "### Model Notes\nBackend and extracted signal details will appear here.",
    )


def _summary_html(result) -> str:
    issue = escape(result.issue_label.title())
    summary = escape(result.summary)
    next_action = escape(result.suggested_next_action)
    return f"""
    <div class="metric-card">
      <div class="summary-title">
        <div>
          <div class="eyebrow">TorchReview Verdict</div>
          <h3 style="margin:0;font-size:30px;">{issue} Issue</h3>
        </div>
        <span class="pill {escape(result.repair_risk)}">{escape(result.repair_risk)} repair risk</span>
      </div>
      <p class="hero-copy">{summary}</p>
        <div class="summary-grid">
        <div class="summary-stat">
          <strong>Reward Score</strong>
          {result.reward_score:.0%}
        </div>
        <div class="summary-stat">
          <strong>ML Quality</strong>
          {result.ml_quality_score:.0%}
        </div>
        <div class="summary-stat">
          <strong>Matched Pattern</strong>
          {escape(result.matched_pattern.title)}
        </div>
        <div class="summary-stat">
          <strong>Inference Backend</strong>
          {escape(result.model_backend)}
        </div>
        <div class="summary-stat">
          <strong>Lint Score</strong>
          {result.lint_score:.0%}
        </div>
        <div class="summary-stat">
          <strong>Complexity Penalty</strong>
          {result.complexity_penalty:.0%}
        </div>
        <div class="summary-stat">
          <strong>Next Action</strong>
          {next_action}
        </div>
      </div>
    </div>
    """


def _radar_html(result) -> str:
    colors = {
        "syntax": "#d95d39",
        "logic": "#4f772d",
        "performance": "#355070",
    }
    bars = []
    for label, score in result.confidence_scores.items():
        bars.append(
            f"""
            <div class="bar">
              <div class="bar-head"><span>{escape(label.title())}</span><span>{score:.0%}</span></div>
              <div class="bar-track">
                <div class="bar-fill" style="width:{score * 100:.1f}%; background:{colors.get(label, '#d95d39')};"></div>
              </div>
            </div>
            """
        )
    return f"""
    <div class="metric-card radar-wrap">
      <div class="eyebrow">Live Triage Radar</div>
      {''.join(bars)}
      <div class="matched-box">
        <strong>Nearest Known Pattern:</strong> {escape(result.matched_pattern.title)}<br>
        <span style="color:#5f6f67;">{escape(result.matched_pattern.summary)}</span>
      </div>
    </div>
    """


def _plan_markdown(result) -> str:
    plan_lines = "\n".join(f"{index + 1}. {step}" for index, step in enumerate(result.repair_plan))
    return (
        "### Improvement Plan\n"
        f"**Primary issue:** `{result.issue_label}`\n\n"
        f"{plan_lines}\n\n"
        f"**Suggested next action:** {result.suggested_next_action}"
    )


def _match_markdown(result) -> str:
    return (
        "### Known Pattern Match\n"
        f"**Task:** `{result.matched_pattern.task_id}`  \n"
        f"**Title:** {result.matched_pattern.title}  \n"
        f"**Why it matched:** {result.matched_pattern.rationale}  \n"
        f"**Similarity:** {result.matched_pattern.similarity:.0%}"
    )


def _model_markdown(result) -> str:
    signal_lines = "\n".join(
        f"- `{signal.name}` -> {signal.value} ({signal.impact}, weight {signal.weight:.2f}): {signal.evidence}"
        for signal in result.extracted_signals
    ) or "- No strong static signals were extracted."
    notes = "\n".join(f"- {item}" for item in result.inference_notes) or "- No additional backend notes."
    return (
        "### Model Notes\n"
        f"- **Model backend:** `{result.model_backend}`\n"
        f"- **Model id:** `{result.model_id}`\n"
        f"- **Analysis time:** `{result.analysis_time_ms:.2f} ms`\n\n"
        "### Reward Formula\n"
        f"- `reward = (0.5 x {result.ml_quality_score:.2f}) + (0.3 x {result.lint_score:.2f}) - (0.2 x {result.complexity_penalty:.2f})`\n"
        f"- **Final reward:** `{result.reward_score:.2f}`\n\n"
        "### Extracted Signals\n"
        f"{signal_lines}\n\n"
        "### Backend Notes\n"
        f"{notes}"
    )


def analyze_inputs(code: str, traceback_text: str, context_window: str) -> tuple[str, str, str, str, str]:
    """Run the triage engine and format outputs for the Gradio UI."""

    result = get_default_engine().triage(code or "", traceback_text or "", context_window or "")
    return (
        _summary_html(result),
        _radar_html(result),
        _plan_markdown(result),
        _match_markdown(result),
        _model_markdown(result),
    )


def load_example(example_key: str) -> tuple[str, str, str, str, str, str, str, str, str]:
    """Populate the UI from a built-in example and immediately analyze it."""

    example = get_default_engine().example_map()[example_key]
    outputs = analyze_inputs(example.code, example.traceback_text, example.context_window)
    header = (
        f"### Example Scenario\n"
        f"**{example.title}**  \n"
        f"{example.summary}  \n"
        f"Label target: `{example.label}`"
    )
    return (example.code, example.traceback_text, example.context_window, header, *outputs)


def build_demo() -> gr.Blocks:
    """Create the TorchReview Copilot Gradio application."""

    examples = get_default_engine().example_map()
    first_example = next(iter(examples.values()))

    with gr.Blocks(title="TorchReview Copilot") as demo:
        gr.HTML(
            """
            <div class="hero-card">
              <div class="eyebrow">Meta PyTorch OpenEnv Hackathon Demo</div>
              <h1 class="hero-title">TorchReview Copilot</h1>
              <p class="hero-copy">
                AI-powered code review and improvement system using PyTorch to score code quality, surface bugs,
                and generate a three-step improvement plan. OpenEnv stays underneath as the deterministic validation engine.
              </p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=6):
                example_choice = gr.Radio(
                    choices=[(item.title, item.key) for item in examples.values()],
                    value=first_example.key,
                    label="Try a built-in failure scenario",
                    info="Switching examples updates the Live Triage Radar immediately.",
                )
                example_header = gr.Markdown()
                code_input = gr.Code(
                    value=first_example.code,
                    language="python",
                    lines=18,
                    label="Python code under review",
                )
                traceback_input = gr.Textbox(
                    value=first_example.traceback_text,
                    lines=7,
                    label="Optional traceback / failing test output",
                    placeholder="Paste stack traces, assertion failures, or benchmark notes here.",
                )
                context_input = gr.Textbox(
                    value=first_example.context_window,
                    lines=4,
                    label="Context window",
                    placeholder="Describe expected behavior, constraints, or repository context.",
                )
                with gr.Row():
                    analyze_button = gr.Button("Analyze & Score Code", variant="primary")
                    clear_button = gr.Button("Clear Inputs", variant="secondary")

            with gr.Column(scale=5):
                summary_html = gr.HTML()
                radar_html = gr.HTML()
                plan_markdown = gr.Markdown()
                match_markdown = gr.Markdown()
                model_markdown = gr.Markdown()

        gr.HTML(
            """
            <div class="subtle-card" style="margin-top: 12px;">
              <div class="eyebrow">How It Works</div>
              <div class="how-grid">
                <div class="how-step"><strong>Input</strong><br>Code plus optional traceback or benchmark signal.</div>
                <div class="how-step"><strong>Processing</strong><br>Static checks extract parser, lint, complexity, and runtime clues.</div>
                <div class="how-step"><strong>Model</strong><br>CodeBERTa embeddings run through PyTorch and score code quality against known OpenEnv patterns.</div>
                <div class="how-step"><strong>Output</strong><br>Confidence radar, reward score, and a three-step improvement plan.</div>
              </div>
            </div>
            """
        )

        example_choice.change(
            fn=load_example,
            inputs=example_choice,
            outputs=[code_input, traceback_input, context_input, example_header, summary_html, radar_html, plan_markdown, match_markdown, model_markdown],
            show_progress="hidden",
        )
        analyze_button.click(
            fn=analyze_inputs,
            inputs=[code_input, traceback_input, context_input],
            outputs=[summary_html, radar_html, plan_markdown, match_markdown, model_markdown],
            show_progress="minimal",
        )
        clear_button.click(
            fn=lambda: ("", "", "", "### Example Scenario\nChoose a built-in example or paste custom code.", *_default_outputs()),
            inputs=None,
            outputs=[code_input, traceback_input, context_input, example_header, summary_html, radar_html, plan_markdown, match_markdown, model_markdown],
            show_progress="hidden",
        )
        demo.load(
            fn=load_example,
            inputs=example_choice,
            outputs=[code_input, traceback_input, context_input, example_header, summary_html, radar_html, plan_markdown, match_markdown, model_markdown],
            show_progress="hidden",
        )

    return demo
