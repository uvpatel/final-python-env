"""Launch the FastAPI backend and Streamlit UI in one Docker container."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    """Start the API backend in the background and keep Streamlit in the foreground."""

    api_process = subprocess.Popen(
        ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"],
    )
    try:
        return subprocess.call(
            [
                "streamlit",
                "run",
                "app/streamlit_app.py",
                "--server.port",
                "8000",
                "--server.address",
                "0.0.0.0",
                "--server.headless",
                "true",
            ]
        )
    finally:
        api_process.terminate()
        api_process.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
