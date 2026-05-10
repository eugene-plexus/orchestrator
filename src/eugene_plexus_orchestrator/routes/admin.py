"""Admin endpoints: /v1/admin/drivers, /v1/admin/nt-state."""

from __future__ import annotations

import asyncio
import logging

import httpx
from fastapi import APIRouter, HTTPException, Request, status

from .._generated.models import (
    BackendKind,
    DriverHealth,
    DriversInfo,
    NTState,
    Problem,
)
from ..bicameral.nt import neutral_state
from ..hemisphere_client import HemisphereClient

router = APIRouter(tags=["admin"])

log = logging.getLogger(__name__)


async def _driver_health(client: HemisphereClient) -> DriverHealth:
    base_url = client.base_url
    try:
        info = await client.info()
        # info.backend is hemisphere-driver.yaml's BackendKind; DriverHealth
        # expects orchestrator.yaml's BackendKind. Same wire values, distinct
        # generated classes — bridge via .value.
        backend = BackendKind(info.backend.value)
        return DriverHealth(
            name=client.name,
            reachable=True,
            url=base_url,  # type: ignore[arg-type]
            backend=backend,
            modelId=info.modelId,
            version=info.version,
        )
    except httpx.HTTPError as e:
        log.warning("driver %r at %s unreachable: %s", client.name, base_url, e)
        return DriverHealth(
            name=client.name,
            reachable=False,
            url=base_url,  # type: ignore[arg-type]
            error=str(e),
        )


@router.get("/v1/admin/drivers", response_model=DriversInfo)
async def list_drivers(request: Request) -> DriversInfo:
    drivers: list[HemisphereClient] = request.app.state.drivers

    if not drivers:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=Problem(
                type="https://github.com/eugene-plexus/orchestrator#no-drivers-configured",
                title="No drivers configured",
                status=503,
                detail=(
                    "The orchestrator has no drivers in its `drivers` config. "
                    "PATCH /v1/config to populate, then restart."
                ),
                component="orchestrator",
            ).model_dump(exclude_none=True),
        )

    healths = await asyncio.gather(*[_driver_health(c) for c in drivers])

    if not any(h.reachable for h in healths):
        summary = "; ".join(f"{h.name}={h.url} ({h.error})" for h in healths)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=Problem(
                type="https://github.com/eugene-plexus/orchestrator#drivers-unreachable",
                title="No drivers reachable",
                status=503,
                detail=f"None of the configured drivers are reachable. {summary}",
                component="orchestrator",
            ).model_dump(exclude_none=True),
        )

    return DriversInfo(drivers=list(healths))


@router.get("/v1/admin/nt-state", response_model=NTState)
async def get_nt_state() -> NTState:
    return neutral_state()
