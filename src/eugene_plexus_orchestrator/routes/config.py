"""Config protocol routes."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from fastapi import APIRouter, Request

from .._generated.models import (
    ConfigDocument,
    ConfigSchema,
    ConfigTestRequest,
    ConfigTestResult,
    ConfigUpdateRequest,
    ConfigUpdateResult,
)
from ..config import ConfigStore, as_schema

router = APIRouter(tags=["config"])


@router.get("/v1/config", response_model=ConfigDocument)
async def get_config(request: Request) -> ConfigDocument:
    store: ConfigStore = request.app.state.config_store
    return store.as_document()


@router.get("/v1/config/schema", response_model=ConfigSchema)
async def get_config_schema() -> ConfigSchema:
    return as_schema()


@router.patch("/v1/config", response_model=ConfigUpdateResult)
async def patch_config(request: Request, body: ConfigUpdateRequest) -> ConfigUpdateResult:
    store: ConfigStore = request.app.state.config_store
    return store.apply_patch(body)


@router.post("/v1/config/test", response_model=ConfigTestResult)
async def test_config(
    request: Request,
    body: ConfigTestRequest | None = None,
) -> ConfigTestResult:
    """Probe every configured driver's `/v1/info` and the memory
    service's `/healthz` using saved config + optional overrides.
    Override values are NOT persisted."""
    start = time.perf_counter()
    store: ConfigStore = request.app.state.config_store

    overrides: dict[str, Any] = {}
    if body and body.overrides:
        overrides = body.overrides.model_dump(exclude_none=True)

    def get(key: str) -> Any:
        return overrides[key] if key in overrides else store.get(key)

    drivers_raw = get("drivers") or []
    memory_url = str(get("memoryUrl") or "")
    timeout = float(get("requestTimeoutSeconds") or 30)

    async def probe(name: str, base_url: str, path: str) -> tuple[str, str | None]:
        """Returns (target-name, error-or-None)."""
        if not base_url:
            return name, f"{name} URL is empty"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(base_url.rstrip("/") + path)
                response.raise_for_status()
        except Exception as e:
            return name, f"{name} ({base_url}{path}) — {e}"
        return name, None

    probes: list[Any] = [probe("memory", memory_url, "/healthz")]
    for entry in drivers_raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "<unnamed>")
        url = str(entry.get("url") or "")
        probes.append(probe(name, url, "/v1/info"))

    results = await asyncio.gather(*probes)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    failures = [err for _, err in results if err]
    if failures:
        return ConfigTestResult(
            ok=False,
            component="orchestrator",
            latencyMs=elapsed_ms,
            error="; ".join(failures),
        )
    driver_count = len(drivers_raw)
    return ConfigTestResult(
        ok=True,
        component="orchestrator",
        latencyMs=elapsed_ms,
        summary=f"all {driver_count} driver(s) + memory reachable in {elapsed_ms}ms",
    )
