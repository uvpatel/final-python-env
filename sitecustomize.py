"""Project-wide UTF-8 console setup for Windows deployments."""

from __future__ import annotations

import sys


def _reconfigure_stream(stream_name: str) -> None:
    stream = getattr(sys, stream_name, None)
    if stream is None:
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_reconfigure_stream("stdout")
_reconfigure_stream("stderr")