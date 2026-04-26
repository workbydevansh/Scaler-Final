"""FastAPI app for the SupportOps-X OpenEnv server."""

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.http_server import serialize_observation
from openenv.core.env_server.types import ResetRequest, ResetResponse, StepRequest, StepResponse
from fastapi import Body, HTTPException
from threading import Lock

from ..models import SupportopsXAction, SupportopsXObservation
from .environment import SupportopsXEnvironment


app = create_app(
    SupportopsXEnvironment,
    SupportopsXAction,
    SupportopsXObservation,
    env_name="supportops_x_env",
    max_concurrent_envs=4,
)

_demo_env = SupportopsXEnvironment()
_demo_lock = Lock()


@app.post(
    "/demo/reset",
    response_model=ResetResponse,
    tags=["Stateful Browser Demo"],
    summary="Reset the stateful browser demo episode",
)
def demo_reset(
    request: ResetRequest = Body(
        default_factory=ResetRequest,
        examples=[
            {"task_id": "enterprise_outage"},
            {"task_id": "account_takeover"},
            {"task_id": "damaged_refund"},
        ],
    ),
) -> ResetResponse:
    """Stateful reset endpoint for Swagger/browser demos.

    The standard OpenEnv `/reset` and `/step` endpoints are kept for validation.
    These `/demo/*` endpoints reuse one in-memory environment so a human can
    click through a multi-step episode in `/docs`.
    """

    kwargs = request.model_dump(exclude_unset=True)
    try:
        with _demo_lock:
            observation = _demo_env.reset(**kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ResetResponse(**serialize_observation(observation))


@app.post(
    "/demo/step",
    response_model=StepResponse,
    tags=["Stateful Browser Demo"],
    summary="Execute one action in the stateful browser demo",
)
def demo_step(request: StepRequest) -> StepResponse:
    """Stateful step endpoint for browser demos."""

    try:
        action = SupportopsXAction(**request.action)
        with _demo_lock:
            observation = _demo_env.step(action, timeout_s=request.timeout_s)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StepResponse(**serialize_observation(observation))


@app.get(
    "/demo/state",
    tags=["Stateful Browser Demo"],
    summary="Inspect the current stateful browser demo state",
)
def demo_state():
    """Return the current state of the stateful browser demo."""

    with _demo_lock:
        return _demo_env.state.model_dump()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the local development server."""

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
