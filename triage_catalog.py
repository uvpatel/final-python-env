"""Curated prototypes and example inputs for TorchReview Copilot."""

from __future__ import annotations

from typing import Dict, List

try:
    from .triage_models import IssueLabel, TriageExample, TriagePrototype
    from .tasks import list_tasks
except ImportError:
    from triage_models import IssueLabel, TriageExample, TriagePrototype
    from tasks import list_tasks


TASK_KIND_TO_LABEL: Dict[str, IssueLabel] = {
    "syntax_fix": "syntax",
    "bug_fix": "logic",
    "optimization": "performance",
}

TRACEBACK_BY_TASK_ID: Dict[str, str] = {
    "syntax_fix_invoice_totals": (
        "Traceback (most recent call last):\n"
        "  File \"services/billing/reconciliation.py\", line 3\n"
        "    for record in records\n"
        "                      ^\n"
        "SyntaxError: expected ':'"
    ),
    "bug_fix_session_windows": (
        "AssertionError: collapse_sessions([{'minute': 1}, {'minute': 3}, {'minute': 8}], 4)\n"
        "Expected: [(1, 3), (8, 8)]\n"
        "Actual:   [(1, 8)]\n"
        "Boundary handling merges the final session instead of starting a new one."
    ),
    "optimization_rank_active_users": (
        "BenchmarkWarning: rank_active_users exceeded the 450ms budget on a nightly export fixture.\n"
        "Profiler hint: repeated scans over the full event list and nested loops dominate runtime."
    ),
}

SUMMARY_BY_TASK_ID: Dict[str, str] = {
    "syntax_fix_invoice_totals": "Broken parser state in a billing helper blocks reconciliation jobs.",
    "bug_fix_session_windows": "Session-boundary logic fails on inclusive idle-timeout edges.",
    "optimization_rank_active_users": "A nightly ranking job is correct on small fixtures but too slow at production scale.",
}

CONTEXT_BY_TASK_ID: Dict[str, str] = {
    "syntax_fix_invoice_totals": (
        "Context window: this helper runs in an end-of-day billing reconciliation job. "
        "Keep the public function signature intact and restore correct totals for mixed integer/string inputs."
    ),
    "bug_fix_session_windows": (
        "Context window: this function groups sorted product analytics events into sessions for retention dashboards. "
        "Boundary behavior must stay deterministic because downstream reports depend on it."
    ),
    "optimization_rank_active_users": (
        "Context window: this pipeline feeds a nightly export on a small CPU instance. "
        "Maintain identical output ordering while improving scalability on larger event volumes."
    ),
}


def _prototype_text(
    task_id: str,
    title: str,
    description: str,
    repo_summary: str,
    goal: str,
    visible_tests: List[str],
    starter_code: str,
    traceback_text: str,
) -> str:
    visible = "\n".join(f"- {item}" for item in visible_tests) or "- none"
    return (
        f"Title: {title}\n"
        f"Problem: {description}\n"
        f"Repo context: {repo_summary}\n"
        f"Goal: {goal}\n"
        f"Observed failure:\n{traceback_text}\n"
        f"Visible checks:\n{visible}\n"
        f"Candidate code:\n{starter_code}\n"
        f"Task id: {task_id}\n"
    )


def build_examples() -> List[TriageExample]:
    """Create stable UI examples from the task catalog."""

    examples: List[TriageExample] = []
    for task in list_tasks():
        label = TASK_KIND_TO_LABEL[task.task_kind]
        examples.append(
            TriageExample(
                key=task.task_id,
                title=task.title,
                label=label,
                summary=SUMMARY_BY_TASK_ID[task.task_id],
                code=task.starter_code,
                traceback_text=TRACEBACK_BY_TASK_ID[task.task_id],
                context_window=CONTEXT_BY_TASK_ID[task.task_id],
                task_id=task.task_id,
            )
        )
    return examples


def build_prototypes() -> List[TriagePrototype]:
    """Build canonical triage prototypes from the OpenEnv tasks."""

    prototypes: List[TriagePrototype] = []
    for task in list_tasks():
        traceback_text = TRACEBACK_BY_TASK_ID[task.task_id]
        prototypes.append(
            TriagePrototype(
                task_id=task.task_id,
                title=task.title,
                label=TASK_KIND_TO_LABEL[task.task_kind],
                summary=SUMMARY_BY_TASK_ID[task.task_id],
                reference_text=_prototype_text(
                    task.task_id,
                    task.title,
                    task.task_description,
                    task.repo_summary,
                    task.goal,
                    list(task.visible_tests),
                    task.reference_code,
                    traceback_text,
                ),
                starter_code=task.starter_code,
                reference_code=task.reference_code,
                traceback_text=traceback_text,
            )
        )
    return prototypes
