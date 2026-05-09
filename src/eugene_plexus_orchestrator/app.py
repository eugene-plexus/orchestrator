"""FastAPI app factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import __version__
from .config import ConfigStore
from .hemisphere_client import HemisphereClient, HttpHemisphereClient
from .memory import InProcessMemory
from .routes import admin as admin_routes
from .routes import chat as chat_routes
from .routes import config as config_routes
from .routes import conversations as conversations_routes
from .routes import health as health_routes
from .settings import Settings, load_settings


def build_clients(store: ConfigStore) -> tuple[HemisphereClient, HemisphereClient, str, str]:
    """Construct left + right hemisphere clients from config.

    Returns (left, right, left_url, right_url). The URLs are returned alongside
    so admin endpoints can include them in HemisphereInfo responses without
    re-reading config.
    """
    left_url = str(store.get("leftDriverUrl"))
    right_url = str(store.get("rightDriverUrl"))
    timeout = float(store.get("requestTimeoutSeconds") or 180)
    left = HttpHemisphereClient(base_url=left_url, timeout_seconds=timeout)
    right = HttpHemisphereClient(base_url=right_url, timeout_seconds=timeout)
    return left, right, left_url, right_url


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    store = ConfigStore(settings.config_file)
    store.load()
    app.state.config_store = store

    app.state.memory = InProcessMemory()

    if not hasattr(app.state, "left_driver") or not hasattr(app.state, "right_driver"):
        # Tests can pre-populate left_driver / right_driver before the lifespan
        # runs to inject fakes. Only build real HTTP clients otherwise.
        left, right, left_url, right_url = build_clients(store)
        app.state.left_driver = left
        app.state.right_driver = right
        app.state.left_driver_url = left_url
        app.state.right_driver_url = right_url
        owns_clients = True
    else:
        owns_clients = False
        app.state.left_driver_url = getattr(app.state, "left_driver_url", "")
        app.state.right_driver_url = getattr(app.state, "right_driver_url", "")

    try:
        yield
    finally:
        if owns_clients:
            await app.state.left_driver.aclose()
            await app.state.right_driver.aclose()


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

    app.include_router(health_routes.router)
    app.include_router(config_routes.router)
    app.include_router(admin_routes.router)
    app.include_router(conversations_routes.router)
    app.include_router(chat_routes.router)

    return app
