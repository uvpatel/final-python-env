"""Client helpers for python_code_review_env."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    PythonCodeReviewAction,
    PythonCodeReviewObservation,
    PythonCodeReviewState,
)


class PythonCodeReviewEnv(
    EnvClient[PythonCodeReviewAction, PythonCodeReviewObservation, PythonCodeReviewState]
):
    """Typed client for the code review environment."""

    def _step_payload(self, action: PythonCodeReviewAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[PythonCodeReviewObservation]:
        observation = PythonCodeReviewObservation.model_validate(payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict) -> PythonCodeReviewState:
        return PythonCodeReviewState.model_validate(payload)


PythonEnv = PythonCodeReviewEnv
