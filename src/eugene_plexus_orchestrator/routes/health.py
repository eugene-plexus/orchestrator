"""GET /healthz."""

from __future__ import annotations

from fastapi import APIRouter

from .. import __version__
from .._generated.models import Health, Status

router = APIRouter(tags=["meta"])


@router.get("/healthz", response_model=Health)
async def healthz() -> Health:
    return Health(
        status=Status.ok,
        version=__version__,
        component="orchestrator",
    )
