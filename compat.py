"""Compatibility helpers expected by validator-oriented scripts."""

from __future__ import annotations


def install_openenv_fastmcp_compat() -> None:
    """Install runtime shims when needed.

    The current environment does not require any monkey-patching, so this is a
    deliberate no-op kept for validator compatibility.
    """

    return None
