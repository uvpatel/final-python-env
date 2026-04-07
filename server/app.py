"""FastAPI entrypoint for python_code_review_env."""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required to run the API server. Install project dependencies first."
    ) from exc

try:
    from ..models import PythonCodeReviewAction, PythonCodeReviewObservation
    from .env import PythonCodeReviewEnvironment
except ImportError:
    from models import PythonCodeReviewAction, PythonCodeReviewObservation
    from server.env import PythonCodeReviewEnvironment


app = create_app(
    PythonCodeReviewEnvironment,
    PythonCodeReviewAction,
    PythonCodeReviewObservation,
    env_name="python_code_review_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
