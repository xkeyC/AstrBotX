"""local-multimodal-infra settings routes for the WebUI (dashboard only)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, Request

from astrbot.dashboard.responses import ApiError, ok
from astrbot.dashboard.services.local_infra_service import (
    LocalInfraService,
    LocalInfraServiceError,
)

from .auth import require_dashboard_user

legacy_router = APIRouter(
    prefix="/api/local_infra",
    tags=["Dashboard local-multimodal-infra"],
    include_in_schema=False,
)


def get_service(request: Request) -> LocalInfraService:
    return request.app.state.services.local_infra


async def _run(coro):
    try:
        return await coro
    except LocalInfraServiceError as exc:
        raise ApiError(str(exc)) from exc


@legacy_router.get("")
async def get_state(
    _username: str = Depends(require_dashboard_user),
    service: LocalInfraService = Depends(get_service),
):
    return ok(service.state())


@legacy_router.post("")
async def save_settings(
    payload: dict[str, Any] = Body(...),
    _username: str = Depends(require_dashboard_user),
    service: LocalInfraService = Depends(get_service),
):
    return ok(await _run(service.save(payload)), message="已保存")


@legacy_router.post("/refresh")
async def refresh(
    _username: str = Depends(require_dashboard_user),
    service: LocalInfraService = Depends(get_service),
):
    return ok(await _run(service.refresh()))


@legacy_router.post("/probe")
async def probe(
    payload: dict[str, Any] = Body(...),
    _username: str = Depends(require_dashboard_user),
    service: LocalInfraService = Depends(get_service),
):
    return ok(await _run(service.probe(payload)))
