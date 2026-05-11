"""HTTP client for talking to a single hemisphere-driver instance.

Implemented as a thin wrapper around an httpx.AsyncClient: one client per
driver, lifetime managed by the FastAPI lifespan. The orchestrator calls
all configured drivers in parallel via asyncio.gather. Each client carries
the operator-supplied driver `name` so the bicameral loop can stamp it
onto every emitted message and the admin endpoint can label it.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from ._generated.hemisphere_models import (
    DriverInfo,
    GenerateRequest,
    GenerateResponse,
    Problem,
)


class HemisphereDriverError(Exception):
    """Raised when a hemisphere-driver responds with 4xx/5xx.

    Carries the driver's parsed `Problem` body (when present) so the
    chat route can surface the *actual* upstream error instead of the
    generic "502 Bad Gateway" httpx text. Drivers return Problem JSON
    via FastAPI's `HTTPException(detail=Problem(...).model_dump())`,
    which produces a `{"detail": {...}}` envelope — we look inside.
    """

    def __init__(
        self,
        *,
        driver_name: str,
        driver_url: str,
        status_code: int,
        problem: Problem | None,
        raw_body: str,
    ) -> None:
        self.driver_name = driver_name
        self.driver_url = driver_url
        self.status_code = status_code
        self.problem = problem
        self.raw_body = raw_body
        super().__init__(self._summary())

    def _summary(self) -> str:
        prefix = f"driver {self.driver_name!r} ({self.driver_url}) returned {self.status_code}"
        if self.problem is not None:
            parts = [prefix, self.problem.title]
            if self.problem.detail:
                parts.append(self.problem.detail)
            if self.problem.component:
                parts.append(f"component={self.problem.component}")
            return " — ".join(parts)
        snippet = self.raw_body[:300] if self.raw_body else "<empty body>"
        return f"{prefix} (no problem+json body): {snippet}"


def _problem_from_response(response: httpx.Response) -> Problem | None:
    """Best-effort extraction of a Problem from a driver's error response.

    Drivers return either:
        a) a bare Problem JSON: `{"type": ..., "title": ..., ...}`
        b) FastAPI's HTTPException-wrapped form: `{"detail": {<problem>}}`

    Try (b) first (the common case), fall back to (a). Any parse failure
    returns None — the caller falls back to the raw body.
    """
    try:
        body: Any = response.json()
    except ValueError:
        return None
    candidates: list[Any] = []
    if isinstance(body, dict):
        if isinstance(body.get("detail"), dict):
            candidates.append(body["detail"])
        candidates.append(body)
    for candidate in candidates:
        try:
            return Problem.model_validate(candidate)
        except ValidationError:
            continue
    return None


class HemisphereClient(Protocol):
    """Contract every hemisphere client implements (real or fake-for-tests)."""

    name: str
    base_url: str

    async def info(self) -> DriverInfo: ...
    async def generate(self, request: GenerateRequest) -> GenerateResponse: ...
    async def aclose(self) -> None: ...


class HttpHemisphereClient:
    """Real HTTP-backed client. Talks to a hemisphere-driver over its OpenAPI."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        timeout_seconds: float = 180.0,
        service_token: str | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        # When the watchdog threaded a service token in, attach it to
        # every outbound call. The driver validates against the shared
        # HMAC signing key. Headers stay unset when running unauthenticated
        # (dev / standalone) so the existing test path still works.
        headers = {"Authorization": f"Bearer {service_token}"} if service_token else None
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            headers=headers,
        )

    async def info(self) -> DriverInfo:
        response = await self._client.get("/v1/info")
        response.raise_for_status()
        return DriverInfo.model_validate(response.json())

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        payload = request.model_dump(mode="json", exclude_none=True)
        response = await self._client.post("/v1/generate", json=payload)
        if response.status_code >= 400:
            raise HemisphereDriverError(
                driver_name=self.name,
                driver_url=self.base_url,
                status_code=response.status_code,
                problem=_problem_from_response(response),
                raw_body=response.text,
            )
        return GenerateResponse.model_validate(response.json())

    async def aclose(self) -> None:
        await self._client.aclose()
