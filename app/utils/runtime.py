"""Formatting, parsing, and IO-suppression helpers for inference."""

from __future__ import annotations

import io
from collections.abc import Iterable
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Iterator

try:
    from tasks import task_ids
except ImportError:  # pragma: no cover
    from python_env.tasks import task_ids  # type: ignore[no-redef]


def compact_text(
    value: Any,
    *,
    default: str = "",
    limit: int = 240,
    preserve_newlines: bool = False,
) -> str:
    """Convert values into validator-safe text."""

    if value is None:
        return default
    try:
        text = str(value)
    except Exception:
        return default
    if preserve_newlines:
        text = text.strip()
    else:
        text = " ".join(text.split())
    return text[:limit] if text else default


def observation_attr(observation: Any, name: str, default: Any = None, *, preserve_newlines: bool = False) -> Any:
    """Read an observation attribute without trusting the payload shape."""

    if isinstance(observation, dict):
        value = observation.get(name, default)
    else:
        value = getattr(observation, name, default)
    if isinstance(value, str):
        return compact_text(
            value,
            default=default if isinstance(default, str) else "",
            preserve_newlines=preserve_newlines,
        )
    return value


def format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def format_reward(value: Any) -> str:
    try:
        reward = float(value)
    except Exception:
        reward = 0.0
    return f"{reward:.2f}"


def format_error(value: Any) -> str:
    text = compact_text(value, default="")
    return text if text else "null"


def parse_task_ids() -> list[str]:
    """Load stable task names with a deterministic fallback."""

    try:
        values = task_ids()
        if isinstance(values, Iterable):
            loaded = [compact_text(item, default="") for item in values]
            loaded = [item for item in loaded if item]
            if loaded:
                return loaded
    except Exception:
        pass
    return [
        "syntax_fix_invoice_totals",
        "bug_fix_session_windows",
        "optimization_rank_active_users",
    ]


@contextmanager
def suppress_output() -> Iterator[None]:
    """Silence libraries that write noisy logs to stdout or stderr."""

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        yield
