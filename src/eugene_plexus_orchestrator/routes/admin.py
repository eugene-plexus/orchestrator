"""Admin endpoints: /v1/admin/hemispheres, /v1/admin/nt-state."""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException, Request, status

from .._generated.models import (
    BackendKind,
    HemisphereInfo,
    HemispherePairInfo,
    NTState,
    Problem,
)
from ..bicameral.nt import neutral_state
from ..hemisphere_client import HemisphereClient

router = APIRouter(tags=["admin"])

log = logging.getLogger(__name__)


async def _hemisphere_info(client: HemisphereClient, base_url: str) -> HemisphereInfo:
    try:
        info = await client.info()
        # info.backend is hemisphere-driver.yaml's BackendKind; HemisphereInfo
        # expects orchestrator.yaml's BackendKind. Same wire values, distinct
        # generated classes — bridge via .value.
        backend = BackendKind(info.backend.value)
        # TODO(specs): HemisphereInfo.url and similar `format: uri` fields make
        # codegen produce AnyUrl. Fold into the next spec polish PR alongside
        # the Problem.type fix that already shipped.
        return HemisphereInfo(
            reachable=True,
            url=base_url,  # type: ignore[arg-type]
            backend=backend,
            modelId=info.modelId,
            version=info.version,
        )
    except httpx.HTTPError as e:
        log.warning("hemisphere %s unreachable: %s", base_url, e)
        return HemisphereInfo(
            reachable=False,
            url=base_url,  # type: ignore[arg-type]
            error=str(e),
        )


@router.get("/v1/admin/hemispheres", response_model=HemispherePairInfo)
async def list_hemispheres(request: Request) -> HemispherePairInfo:
    left: HemisphereClient = request.app.state.left_driver
    right: HemisphereClient = request.app.state.right_driver
    left_url: str = request.app.state.left_driver_url
    right_url: str = request.app.state.right_driver_url

    left_info = await _hemisphere_info(left, left_url)
    right_info = await _hemisphere_info(right, right_url)

    pair = HemispherePairInfo(left=left_info, right=right_info)

    if not left_info.reachable and not right_info.reachable:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=Problem(
                type="https://github.com/eugene-plexus/orchestrator#hemispheres-unreachable",
                title="Both hemispheres unreachable",
                status=503,
                detail=(
                    f"Neither hemisphere-driver is reachable. "
                    f"left={left_url} ({left_info.error}); "
                    f"right={right_url} ({right_info.error})."
                ),
                component="orchestrator",
            ).model_dump(exclude_none=True),
        )

    return pair


@router.get("/v1/admin/nt-state", response_model=NTState)
async def get_nt_state() -> NTState:
    return neutral_state()
