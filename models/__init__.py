"""PyTorch-backed model wrappers plus OpenEnv schema exports."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pytorch_model import PyTorchCodeAnalyzerModel


def _load_schema_module():
    schema_path = Path(__file__).resolve().parent.parent / "models.py"
    spec = importlib.util.spec_from_file_location("_python_env_schema_models", schema_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"Unable to load schema models from {schema_path}")
    if spec.name in sys.modules:
        return sys.modules[spec.name]
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for model_name in (
        "HistoryEntry",
        "RewardDetails",
        "PythonCodeReviewAction",
        "PythonCodeReviewObservation",
        "PythonCodeReviewState",
        "TaskDescriptor",
        "TaskSummary",
        "TaskGrade",
        "HealthResponse",
    ):
        getattr(module, model_name).model_rebuild()
    return module


_schema_models = _load_schema_module()

HealthResponse = _schema_models.HealthResponse
HistoryEntry = _schema_models.HistoryEntry
PythonAction = _schema_models.PythonAction
PythonCodeReviewAction = _schema_models.PythonCodeReviewAction
PythonCodeReviewObservation = _schema_models.PythonCodeReviewObservation
PythonCodeReviewState = _schema_models.PythonCodeReviewState
PythonObservation = _schema_models.PythonObservation
PythonState = _schema_models.PythonState
RewardDetails = _schema_models.RewardDetails
TaskDescriptor = _schema_models.TaskDescriptor
TaskGrade = _schema_models.TaskGrade
TaskSummary = _schema_models.TaskSummary


def __getattr__(name: str):
    if name == "PyTorchCodeAnalyzerModel":
        from .pytorch_model import PyTorchCodeAnalyzerModel as model_class

        return model_class
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "HealthResponse",
    "HistoryEntry",
    "PyTorchCodeAnalyzerModel",
    "PythonAction",
    "PythonCodeReviewAction",
    "PythonCodeReviewObservation",
    "PythonCodeReviewState",
    "PythonObservation",
    "PythonState",
    "RewardDetails",
    "TaskDescriptor",
    "TaskGrade",
    "TaskSummary",
]
