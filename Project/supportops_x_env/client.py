"""OpenEnv client for SupportOps-X."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import SupportopsXAction, SupportopsXObservation, SupportopsXState


class SupportopsXEnv(
    EnvClient[SupportopsXAction, SupportopsXObservation, SupportopsXState]
):
    """Async client for a running SupportOps-X environment server."""

    def _step_payload(self, action: SupportopsXAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SupportopsXObservation]:
        obs_data = payload.get("observation", {})
        observation = SupportopsXObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SupportopsXState:
        return SupportopsXState(**payload)

