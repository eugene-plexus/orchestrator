"""Pytest fixtures shared across the test suite.

We stub the hemisphere clients before the FastAPI lifespan runs so tests
never try to reach real URLs. Each test can replace the per-hemisphere
script of canned responses by mutating the fake clients on the app state.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_orchestrator._generated.hemisphere_models import (
    BackendKind,
    DriverInfo,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    Hemisphere,
)
from eugene_plexus_orchestrator.app import create_app
from eugene_plexus_orchestrator.settings import Settings


class FakeHemisphereClient:
    """In-memory test double implementing HemisphereClient."""

    def __init__(
        self,
        *,
        hemisphere: Hemisphere,
        backend: BackendKind = BackendKind.claude_code_cli,
        model_id: str = "fake-model",
    ):
        self.hemisphere = hemisphere
        self.backend = backend
        self.model_id = model_id
        self.responses: list[str] = []
        """FIFO queue of canned responses; tests assign before calling the route."""
        self.calls: list[GenerateRequest] = []
        self.info_error: Exception | None = None

    async def info(self) -> DriverInfo:
        if self.info_error is not None:
            raise self.info_error
        return DriverInfo(
            backend=self.backend,
            modelId=self.model_id,
            hemisphere=self.hemisphere,
            version="0.0.0-fake",
        )

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.calls.append(request)
        if not self.responses:
            text = f"<{self.hemisphere.value} default response>"
        else:
            text = self.responses.pop(0)
        return GenerateResponse(
            content=text,
            finishReason=FinishReason.stop,
            backend=self.backend,
            modelId=self.model_id,
            latencyMs=1,
        )

    async def aclose(self) -> None:
        return None


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(config_file=tmp_path / "config.yaml")


@pytest.fixture
def left_fake() -> FakeHemisphereClient:
    return FakeHemisphereClient(hemisphere=Hemisphere.left)


@pytest.fixture
def right_fake() -> FakeHemisphereClient:
    return FakeHemisphereClient(
        hemisphere=Hemisphere.right, backend=BackendKind.codex_cli, model_id="fake-gpt"
    )


@pytest.fixture
def app(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> FastAPI:
    app = create_app(settings=settings)
    app.state.left_driver = left_fake
    app.state.right_driver = right_fake
    app.state.left_driver_url = "http://fake-left"
    app.state.right_driver_url = "http://fake-right"
    return app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
