"""HTTP client for talking to a single hemisphere-driver instance.

Implemented as a thin wrapper around an httpx.AsyncClient: one client per
hemisphere, lifetime managed by the FastAPI lifespan. The orchestrator
calls both hemispheres in parallel via asyncio.gather.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from ._generated.hemisphere_models import (
    DriverInfo,
    GenerateRequest,
    GenerateResponse,
)


class HemisphereClient(Protocol):
    """Contract every hemisphere client implements (real or fake-for-tests)."""

    async def info(self) -> DriverInfo: ...
    async def generate(self, request: GenerateRequest) -> GenerateResponse: ...
    async def aclose(self) -> None: ...


class HttpHemisphereClient:
    """Real HTTP-backed client. Talks to a hemisphere-driver over its OpenAPI."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 180.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    async def info(self) -> DriverInfo:
        response = await self._client.get("/v1/info")
        response.raise_for_status()
        return DriverInfo.model_validate(response.json())

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        payload = request.model_dump(mode="json", exclude_none=True)
        response = await self._client.post("/v1/generate", json=payload)
        response.raise_for_status()
        return GenerateResponse.model_validate(response.json())

    async def aclose(self) -> None:
        await self._client.aclose()
