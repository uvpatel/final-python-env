"""OpenEnv FastAPI entrypoint with optional Gradio mounting."""

from __future__ import annotations

import os

from fastapi import FastAPI

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
except ImportError:
    from models import PythonCodeReviewAction, PythonCodeReviewObservation
    from server.env import PythonCodeReviewEnvironment


def _gradio_enabled() -> bool:
    for env_name in ("ENABLE_GRADIO_DEMO", "ENABLE_WEB_INTERFACE"):
        if str(os.getenv(env_name, "")).strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def _max_concurrent_envs() -> int:
    try:
        return max(int(os.getenv("OPENENV_MAX_CONCURRENT_ENVS", "2")), 1)
    except Exception:
        return 2


def build_application():
    """Compose the OpenEnv API with the Gradio demo frontend."""

    api_app = create_app(
        PythonCodeReviewEnvironment,
        PythonCodeReviewAction,
        PythonCodeReviewObservation,
        env_name="python_code_review_env",
        max_concurrent_envs=_max_concurrent_envs(),
    )
    served_app = api_app
    if gr is not None and _gradio_enabled():
        try:
            from .demo import CSS, build_demo
        except ImportError:
            from server.demo import CSS, build_demo
        served_app = gr.mount_gradio_app(
            api_app,
            build_demo(),
            path="/",
            theme=gr.themes.Soft(primary_hue="orange", secondary_hue="amber"),
            css=CSS,
        )

    wrapper_app = FastAPI(title="python_code_review_env", version="1.0.0")

    @wrapper_app.get("/health", include_in_schema=False)
    def _health() -> dict[str, str]:
        return {"status": "ok"}

    wrapper_app.mount("/", served_app)
    return wrapper_app


app = build_application()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, access_log=False)


if __name__ == "__main__":
    main()
