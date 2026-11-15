"""Demo-only fault-injection routes.

HOW IT FITS IN THE SYSTEM:
Mounted by app.py's create_app() but inert unless ATHENA_DEMO_MODE is set:
lifespan then stores `app.state.demo_models` (role -> DemoMockModel) and
`app.state.routed_model`. Without them every route here returns 404, so a
normal deployment exposes nothing.

  GET  /debug/status   per-backend failing flag + circuit-breaker state
  POST /debug/fail     {"role": "fast-mock", "failing": true} break/fix a backend
  POST /debug/reset    heal every backend and close every circuit

These exist so scripts/demo.py (or a person with Swagger open) can show
retry -> circuit open -> fallback without editing code. They deliberately
sit outside require_api_key/rate_limit: demo mode is local-only.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

debug_router = APIRouter(prefix="/debug", tags=["demo"])


class FailRequest(BaseModel):
    role: str
    failing: bool = True


def _state(request: Request):  # type: ignore[no-untyped-def]
    models = getattr(request.app.state, "demo_models", None)
    routed = getattr(request.app.state, "routed_model", None)
    if not models or routed is None:
        raise HTTPException(status_code=404, detail="Demo mode is not enabled")
    return models, routed


def _status(models, routed) -> dict[str, dict[str, object]]:  # type: ignore[no-untyped-def]
    circuits = routed.circuit_states()
    return {
        role: {"failing": model.failing, "circuit": circuits[role]}
        for role, model in models.items()
    }


@debug_router.get("/status")
async def status(request: Request) -> dict[str, dict[str, object]]:
    models, routed = _state(request)
    return _status(models, routed)


@debug_router.post("/fail")
async def fail(body: FailRequest, request: Request) -> dict[str, dict[str, object]]:
    models, routed = _state(request)
    if body.role not in models:
        raise HTTPException(status_code=404, detail=f"Unknown role: {body.role}")
    models[body.role].set_failing(body.failing)
    return _status(models, routed)


@debug_router.post("/reset")
async def reset(request: Request) -> dict[str, dict[str, object]]:
    models, routed = _state(request)
    for model in models.values():
        model.set_failing(False)
    routed.reset_circuits()
    return _status(models, routed)
