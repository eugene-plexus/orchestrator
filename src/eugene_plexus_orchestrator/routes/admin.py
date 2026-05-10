"""Admin endpoints: /v1/admin/drivers, /v1/admin/drivers/probe, /v1/admin/nt-state."""

from __future__ import annotations

import asyncio
import logging

import httpx
from fastapi import APIRouter, HTTPException, Request, status

from .._generated.models import (
    BackendKind,
    DriverEntry,
    DriverHealth,
    DriversInfo,
    NTState,
    Problem,
)
from ..bicameral.nt import neutral_state
from ..hemisphere_client import HemisphereClient, HttpHemisphereClient

router = APIRouter(tags=["admin"])

log = logging.getLogger(__name__)

# Probe timeout: deliberately short. The UI's per-row Test button is an
# interactive affordance — operators want a quick yes/no, not a 3-minute
# wait on a hung URL. Real generation calls use the full
# `requestTimeoutSeconds` from config.
_PROBE_TIMEOUT_SECONDS = 10.0


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


@router.post("/v1/admin/drivers/probe", response_model=DriverHealth)
async def probe_driver(body: DriverEntry) -> DriverHealth:
    """Test-connect to an arbitrary driver URL without persisting it.

    Backs the UI's per-row Test button in the drivers list editor —
    operators verify a URL is reachable before saving the topology.
    Builds a one-shot HTTP client, hits the URL's `/v1/info`, and
    returns the same `DriverHealth` shape the list endpoint uses.
    """
    url = str(body.url).rstrip("/")
    client = HttpHemisphereClient(
        name=body.name,
        base_url=url,
        timeout_seconds=_PROBE_TIMEOUT_SECONDS,
    )
    try:
        return await _driver_health(client)
    finally:
        await client.aclose()


@router.get("/v1/admin/nt-state", response_model=NTState)
async def get_nt_state() -> NTState:
    return neutral_state()
