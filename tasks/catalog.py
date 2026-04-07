"""Deterministic task definitions for the code review environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Dict, List, Optional


def _code(value: str) -> str:
    return dedent(value).strip() + "\n"


@dataclass(frozen=True)
class CallCase:
    """One executable function call used by graders."""

    label: str
    args: tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    expected: Any = None


@dataclass(frozen=True)
class ReviewTask:
    """Static task definition."""

    task_id: str
    title: str
    difficulty: str
    task_kind: str
    task_description: str
    starter_code: str
    reference_code: str
    function_name: str
    visible_tests: List[str]
    public_cases: List[CallCase]
    hidden_cases: List[CallCase]
    repo_summary: str
    changed_files: List[str]
    available_files: List[str]
    goal: str
    max_steps: int
    benchmark_config: Optional[Dict[str, int]] = None


TASKS: List[ReviewTask] = [
    ReviewTask(
        task_id="syntax_fix_invoice_totals",
        title="Fix the invoice total syntax regression",
        difficulty="easy",
        task_kind="syntax_fix",
        task_description=(
            "A recent refactor broke the helper that normalizes invoice totals before "
            "daily reconciliation. Repair the Python syntax so the function compiles "
            "and returns the correct total for mixed integer and string inputs."
        ),
        starter_code=_code(
            """
            def normalize_invoice_totals(records):
                cleaned = []
                for record in records
                    if "total" not in record:
                        continue
                    value = int(record["total"])
                    cleaned.append(value)
                return sum(cleaned
            """
        ),
        reference_code=_code(
            '''
            def normalize_invoice_totals(records):
                """Return the sum of invoice totals that are present in the payload."""
                cleaned = []
                for record in records:
                    if "total" not in record:
                        continue
                    cleaned.append(int(record["total"]))
                return sum(cleaned)
            '''
        ),
        function_name="normalize_invoice_totals",
        visible_tests=[
            "normalize_invoice_totals([{'total': '4'}, {'total': 5}, {}]) == 9",
            "normalize_invoice_totals([]) == 0",
        ],
        public_cases=[
            CallCase(
                label="mixed string and int totals",
                args=([{"total": "4"}, {"total": 5}, {}],),
                expected=9,
            ),
            CallCase(label="empty input", args=([],), expected=0),
        ],
        hidden_cases=[
            CallCase(
                label="skip missing totals",
                args=([{}, {"total": "2"}, {"total": "8"}],),
                expected=10,
            ),
            CallCase(
                label="handle negative adjustments",
                args=([{"total": "11"}, {"total": -3}],),
                expected=8,
            ),
        ],
        repo_summary=(
            "services/billing/reconciliation.py computes end-of-day invoice totals for "
            "a CPU-only batch job."
        ),
        changed_files=["services/billing/reconciliation.py"],
        available_files=["services/billing/reconciliation.py", "tests/test_reconciliation.py"],
        goal="Restore a compiling implementation for invoice total normalization.",
        max_steps=6,
    ),
    ReviewTask(
        task_id="bug_fix_session_windows",
        title="Repair session window collapsing logic",
        difficulty="medium",
        task_kind="bug_fix",
        task_description=(
            "The session aggregator regressed after a cleanup pass. Public tests expose "
            "incorrect boundary handling and the final session is missing. Fix the logic "
            "without changing the function contract."
        ),
        starter_code=_code(
            """
            def collapse_sessions(events, idle_timeout_minutes):
                if not events:
                    return []

                sessions = []
                current_start = events[0]["minute"]
                current_end = current_start

                for event in events[1:]:
                    minute = event["minute"]
                    if minute - current_end > idle_timeout_minutes:
                        sessions.append((current_start, current_end))
                        current_start = minute
                    current_end = minute

                return sessions
            """
        ),
        reference_code=_code(
            '''
            def collapse_sessions(events, idle_timeout_minutes):
                """Collapse activity events into inclusive session windows."""
                if not events:
                    return []

                sessions = []
                current_start = events[0]["minute"]
                current_end = current_start

                for event in events[1:]:
                    minute = event["minute"]
                    if minute - current_end >= idle_timeout_minutes:
                        sessions.append((current_start, current_end))
                        current_start = minute
                    current_end = minute

                sessions.append((current_start, current_end))
                return sessions
            '''
        ),
        function_name="collapse_sessions",
        visible_tests=[
            "collapse_sessions([{'minute': 1}, {'minute': 3}, {'minute': 8}], 4) == [(1, 3), (8, 8)]",
            "collapse_sessions([{'minute': 5}, {'minute': 9}], 4) == [(5, 5), (9, 9)]",
        ],
        public_cases=[
            CallCase(
                label="split when idle timeout is exceeded",
                args=([{"minute": 1}, {"minute": 3}, {"minute": 8}], 4),
                expected=[(1, 3), (8, 8)],
            ),
            CallCase(
                label="boundary is inclusive",
                args=([{"minute": 5}, {"minute": 9}], 4),
                expected=[(5, 5), (9, 9)],
            ),
        ],
        hidden_cases=[
            CallCase(
                label="single continuous session",
                args=([{"minute": 2}, {"minute": 4}, {"minute": 5}], 4),
                expected=[(2, 5)],
            ),
            CallCase(label="empty input", args=([], 10), expected=[]),
            CallCase(
                label="multiple boundaries",
                args=([{"minute": 1}, {"minute": 5}, {"minute": 9}, {"minute": 14}], 4),
                expected=[(1, 1), (5, 5), (9, 9), (14, 14)],
            ),
        ],
        repo_summary=(
            "analytics/sessionizer.py condenses sorted clickstream events into user "
            "sessions for downstream retention reports."
        ),
        changed_files=["analytics/sessionizer.py"],
        available_files=["analytics/sessionizer.py", "tests/test_sessionizer.py"],
        goal="Make session collapsing match the expected timeout semantics.",
        max_steps=8,
    ),
    ReviewTask(
        task_id="optimization_rank_active_users",
        title="Optimize the active-user ranking pipeline",
        difficulty="hard",
        task_kind="optimization",
        task_description=(
            "The reporting job is correct enough for small fixtures but too slow for the "
            "daily production export. Preserve the API, keep the output deterministic, "
            "and refactor the implementation for speed and readability."
        ),
        starter_code=_code(
            """
            def rank_active_users(events):
                users = []
                for event in events:
                    if event["status"] == "active":
                        found = False
                        for existing in users:
                            if existing == event["user_id"]:
                                found = True
                        if not found:
                            users.append(event["user_id"])

                totals = []
                for user in users:
                    count = 0
                    for event in events:
                        if event["status"] == "active" and event["user_id"] == user:
                            count = count + 1
                    totals.append((user, count))

                totals.sort(key=lambda item: (-item[1], item[0]))
                return totals
            """
        ),
        reference_code=_code(
            '''
            from collections import Counter


            def rank_active_users(events):
                """Return users ranked by number of active events."""
                counts = Counter(
                    event["user_id"]
                    for event in events
                    if event["status"] == "active"
                )
                return sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            '''
        ),
        function_name="rank_active_users",
        visible_tests=[
            "rank_active_users([{'user_id': 'b', 'status': 'active'}, {'user_id': 'a', 'status': 'active'}, {'user_id': 'b', 'status': 'inactive'}]) == [('a', 1), ('b', 1)]",
            "rank_active_users([{'user_id': 'u1', 'status': 'active'}, {'user_id': 'u1', 'status': 'active'}, {'user_id': 'u2', 'status': 'active'}]) == [('u1', 2), ('u2', 1)]",
        ],
        public_cases=[
            CallCase(
                label="inactive events are ignored",
                args=([{"user_id": "b", "status": "active"}, {"user_id": "a", "status": "active"}, {"user_id": "b", "status": "inactive"}],),
                expected=[("a", 1), ("b", 1)],
            ),
            CallCase(
                label="counts repeated active users",
                args=([{"user_id": "u1", "status": "active"}, {"user_id": "u1", "status": "active"}, {"user_id": "u2", "status": "active"}],),
                expected=[("u1", 2), ("u2", 1)],
            ),
        ],
        hidden_cases=[
            CallCase(
                label="stable alphabetical tie-break",
                args=([{"user_id": "u3", "status": "active"}, {"user_id": "u2", "status": "active"}, {"user_id": "u3", "status": "active"}, {"user_id": "u2", "status": "active"}],),
                expected=[("u2", 2), ("u3", 2)],
            ),
            CallCase(label="empty input", args=([],), expected=[]),
            CallCase(
                label="mixed active and inactive states",
                args=([{"user_id": "x", "status": "inactive"}, {"user_id": "x", "status": "active"}, {"user_id": "y", "status": "active"}, {"user_id": "x", "status": "active"}],),
                expected=[("x", 2), ("y", 1)],
            ),
        ],
        repo_summary=(
            "reports/activity_rankings.py feeds a nightly export that runs on a small CPU "
            "instance and has become too slow after customer growth."
        ),
        changed_files=["reports/activity_rankings.py"],
        available_files=["reports/activity_rankings.py", "tests/test_activity_rankings.py"],
        goal="Keep the output stable while improving runtime and code quality.",
        max_steps=10,
        benchmark_config={"user_pool": 240, "events_per_user": 36, "iterations": 8},
    ),
]

TASK_BY_ID = {task.task_id: task for task in TASKS}


def list_tasks() -> List[ReviewTask]:
    """Return all supported tasks."""

    return list(TASKS)


def get_task(task_id: str) -> ReviewTask:
    """Fetch a task by identifier."""

    try:
        return TASK_BY_ID[task_id]
    except KeyError as exc:
        raise ValueError(f"Unknown task_id: {task_id}") from exc


def select_task(seed: Optional[int] = None, task_id: Optional[str] = None) -> ReviewTask:
    """Select a task deterministically by explicit id or seed."""

    if task_id:
        return get_task(task_id)
    if seed is None:
        return TASKS[0]
    return TASKS[seed % len(TASKS)]
