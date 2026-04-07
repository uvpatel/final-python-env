"""Task catalog for python_code_review_env."""

from .catalog import ReviewTask, get_task, list_tasks, select_task


def task_ids() -> list[str]:
    """Return stable task identifiers for validators."""

    return [task.task_id for task in list_tasks()]


__all__ = ["ReviewTask", "get_task", "list_tasks", "select_task", "task_ids"]
