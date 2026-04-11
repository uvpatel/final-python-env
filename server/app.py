"""FastAPI + Gradio entrypoint for TorchReview Copilot."""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required to run the API server. Install project dependencies first."
    ) from exc

try:
    import gradio as gr
except Exception:
    gr = None  # type: ignore[assignment]

try:
    from ..models import PythonCodeReviewAction, PythonCodeReviewObservation
    from .env import PythonCodeReviewEnvironment
    from .demo import build_demo
except ImportError:
    from models import PythonCodeReviewAction, PythonCodeReviewObservation
    from server.env import PythonCodeReviewEnvironment
    from server.demo import build_demo


def build_application():
    """Compose the OpenEnv API with the Gradio demo frontend."""

    api_app = create_app(
        PythonCodeReviewEnvironment,
        PythonCodeReviewAction,
        PythonCodeReviewObservation,
        env_name="python_code_review_env",
        max_concurrent_envs=4,
    )
    if gr is None:
        return api_app
    return gr.mount_gradio_app(api_app, build_demo(), path="/")


app = build_application()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
