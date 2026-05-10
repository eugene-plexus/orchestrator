"""Pytest fixtures shared across the test suite.

We stub the hemisphere clients before the FastAPI lifespan runs so tests
never try to reach real URLs. Each test can replace the per-driver
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
)
from eugene_plexus_orchestrator.app import create_app
from eugene_plexus_orchestrator.memory import InProcessMemory
from eugene_plexus_orchestrator.settings import Settings


class FakeHemisphereClient:
    """In-memory test double implementing HemisphereClient."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str = "http://fake-driver",
        backend: BackendKind = BackendKind.claude_code_cli,
        model_id: str = "fake-model",
    ):
        self.name = name
        self.base_url = base_url
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
            version="0.0.0-fake",
        )

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.calls.append(request)
        text = self.responses.pop(0) if self.responses else f"<{self.name} default response>"
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
    return FakeHemisphereClient(name="left", base_url="http://fake-left")


@pytest.fixture
def right_fake() -> FakeHemisphereClient:
    return FakeHemisphereClient(
        name="right",
        base_url="http://fake-right",
        backend=BackendKind.codex_cli,
        model_id="fake-gpt",
    )


@pytest.fixture
def app(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> FastAPI:
    app = create_app(settings=settings)
    app.state.drivers = [left_fake, right_fake]
    app.state.memory = InProcessMemory()
    app.state.memory_url = "in-process"
    return app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
