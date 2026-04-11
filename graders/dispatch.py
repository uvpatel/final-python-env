"""Task grader dispatch."""

from __future__ import annotations

try:
    from ..openenv_models import TaskGrade
    from ..tasks.catalog import ReviewTask
except ImportError:
    from openenv_models import TaskGrade
    from tasks.catalog import ReviewTask

from .bug_fix import grade_bug_fix_task
from .optimization import grade_optimization_task
from .syntax import grade_syntax_task


def grade_task(
    task: ReviewTask,
    code: str,
    *,
    include_hidden: bool,
    timeout_s: float = 3.0,
) -> TaskGrade:
    """Dispatch to the correct deterministic grader."""

    if task.task_kind == "syntax_fix":
        return grade_syntax_task(task, code, timeout_s=timeout_s)
    if task.task_kind == "bug_fix":
        return grade_bug_fix_task(task, code, include_hidden=include_hidden, timeout_s=timeout_s)
    if task.task_kind == "optimization":
        return grade_optimization_task(task, code, include_hidden=include_hidden, timeout_s=timeout_s)
    raise ValueError(f"Unsupported task kind: {task.task_kind}")
