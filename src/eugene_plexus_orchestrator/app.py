"""FastAPI app factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from . import __version__
from .config import ConfigStore
from .hemisphere_client import HemisphereClient, HttpHemisphereClient
from .memory import HttpMemory, MemoryClient
from .routes import admin as admin_routes
from .routes import chat as chat_routes
from .routes import config as config_routes
from .routes import conversations as conversations_routes
from .routes import health as health_routes
from .settings import Settings, load_settings


def build_clients(store: ConfigStore) -> list[HemisphereClient]:
    """Construct one HemisphereClient per configured driver.

    Reads the `drivers` config field — a list of {name, url} entries — and
    builds an `HttpHemisphereClient` for each. Order is preserved so the
    bicameral loop and admin endpoints walk the drivers in the operator's
    declared order.
    """
    raw = store.get("drivers") or []
    timeout = float(store.get("requestTimeoutSeconds") or 180)
    clients: list[HemisphereClient] = []
    for entry in raw:
        # Validation in ConfigStore.apply_patch already enforces shape, but
        # in-memory config can also be loaded straight from YAML so guard
        # again here.
        name = entry["name"]
        url = entry["url"]
        clients.append(HttpHemisphereClient(name=name, base_url=url, timeout_seconds=timeout))
    return clients


def build_memory(store: ConfigStore) -> tuple[MemoryClient, str]:
    """Construct the memory client from config. Returns (client, url)."""
    memory_url = str(store.get("memoryUrl") or "http://127.0.0.1:8083")
    # Memory ops are short — append a message, fetch a conversation. The
    # 30s ceiling here is far more generous than needed but matches the
    # connection-timeout shape used elsewhere; bump only if a future
    # backend (DB-backed retrieval, vector search) needs longer.
    timeout = 30.0
    return HttpMemory(base_url=memory_url, timeout_seconds=timeout), memory_url


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    store = ConfigStore(settings.config_file)
    store.load()
    app.state.config_store = store

    # Same injection trick as hemisphere clients: tests pre-populate
    # `app.state.memory` with an `InProcessMemory` so the lifespan
    # doesn't try to reach a real memory service. In production the
    # `HttpMemory` is built here.
    if not hasattr(app.state, "memory"):
        memory, memory_url = build_memory(store)
        app.state.memory = memory
        app.state.memory_url = memory_url
        owns_memory = True
    else:
        owns_memory = False
        app.state.memory_url = getattr(app.state, "memory_url", "")

    if not hasattr(app.state, "drivers"):
        app.state.drivers = build_clients(store)
        owns_clients = True
    else:
        owns_clients = False

    try:
        yield
    finally:
        if owns_memory:
            await app.state.memory.aclose()
        if owns_clients:
            for client in app.state.drivers:
                await client.aclose()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a FastAPI app with all routers mounted."""
    settings = settings or load_settings()

    app = FastAPI(
        title="Eugene Plexus — orchestrator",
        description="Bicameral chat orchestrator.",
        version=__version__,
        lifespan=_lifespan,
    )
    app.state.settings = settings
    # `drivers` is the canonical list of hemisphere clients in declared
    # order. Tests inject this on `app.state` before the lifespan runs;
    # the lifespan otherwise builds it from `ConfigStore`.
    _: Any = app.state  # placate mypy on dynamic state access below

    app.include_router(health_routes.router)
    app.include_router(config_routes.router)
    app.include_router(admin_routes.router)
    app.include_router(conversations_routes.router)
    app.include_router(chat_routes.router)

    return app
