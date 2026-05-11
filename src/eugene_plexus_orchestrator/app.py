"""FastAPI app factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI

from . import __version__
from .auth_state import AuthState, load_auth_state
from .config import ConfigStore
from .dependencies import require_authorized, require_operator
from .hemisphere_client import HemisphereClient, HttpHemisphereClient
from .memory import HttpMemory, MemoryClient
from .routes import admin as admin_routes
from .routes import chat as chat_routes
from .routes import config as config_routes
from .routes import conversations as conversations_routes
from .routes import health as health_routes
from .settings import Settings, load_settings

log = logging.getLogger(__name__)


def build_clients(store: ConfigStore, auth_state: AuthState) -> list[HemisphereClient]:
    """Construct one HemisphereClient per configured driver.

    Reads the `drivers` config field — a list of {name, url} entries — and
    builds an `HttpHemisphereClient` for each. Order is preserved so the
    bicameral loop and admin endpoints walk the drivers in the operator's
    declared order. Threads `auth_state.service_token` through so each
    outbound /v1/generate carries the orchestrator's service-audience
    bearer; the driver validates that against the shared signing key.
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
        clients.append(
            HttpHemisphereClient(
                name=name,
                base_url=url,
                timeout_seconds=timeout,
                service_token=auth_state.service_token,
            )
        )
    return clients


def build_memory(store: ConfigStore, auth_state: AuthState) -> tuple[MemoryClient, str]:
    """Construct the memory client from config. Returns (client, url)."""
    memory_url = str(store.get("memoryUrl") or "http://127.0.0.1:8083")
    # Memory ops are short — append a message, fetch a conversation. The
    # 30s ceiling here is far more generous than needed but matches the
    # connection-timeout shape used elsewhere; bump only if a future
    # backend (DB-backed retrieval, vector search) needs longer.
    timeout = 30.0
    return (
        HttpMemory(
            base_url=memory_url,
            timeout_seconds=timeout,
            service_token=auth_state.service_token,
        ),
        memory_url,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    store = ConfigStore(settings.config_file)
    if settings.safe_mode:
        # Safe mode: ignore the on-disk config, leaving the store with
        # built-in defaults. PATCH /v1/config still writes to disk so the
        # operator's repair survives the next non-safe-mode boot. No
        # drivers, no memory client — chat returns 503 until restart.
        log.warning(
            "starting in SAFE MODE (EUGENE_PLEXUS_ORCH_SAFE_MODE=1); "
            "ignoring %s and running on defaults. Fix config via "
            "/v1/config, then restart without the env var.",
            settings.config_file,
        )
    else:
        store.load()
    app.state.config_store = store
    app.state.safe_mode = settings.safe_mode

    # v0.2 auth state. Tests can pre-populate `app.state.auth_state` to
    # exercise authed paths; the default lifespan build reads env vars
    # via Settings and produces an auth-disabled state when the watchdog
    # didn't supply AUTH_SIGNING_KEY.
    if not hasattr(app.state, "auth_state"):
        app.state.auth_state = load_auth_state(
            signing_key_b64=settings.auth_signing_key,
            service_token=settings.service_token,
            master_key_b64=settings.master_key,
        )
    auth_state: AuthState = app.state.auth_state

    # Same injection trick as hemisphere clients: tests pre-populate
    # `app.state.memory` with an `InProcessMemory` so the lifespan
    # doesn't try to reach a real memory service. In production the
    # `HttpMemory` is built here.
    if not hasattr(app.state, "memory"):
        if settings.safe_mode:
            app.state.memory = None
            app.state.memory_url = ""
            owns_memory = False
        else:
            memory, memory_url = build_memory(store, auth_state)
            app.state.memory = memory
            app.state.memory_url = memory_url
            owns_memory = True
    else:
        owns_memory = False
        app.state.memory_url = getattr(app.state, "memory_url", "")

    if not hasattr(app.state, "drivers"):
        # Defaults have no `drivers` configured; build_clients returns []
        # in safe mode anyway, but skip the call for clarity.
        app.state.drivers = [] if settings.safe_mode else build_clients(store, auth_state)
        owns_clients = not settings.safe_mode
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

    # Health stays unauthenticated — supervisors and load balancers need
    # to probe it without holding credentials.
    app.include_router(health_routes.router)

    # Operator-only surfaces: config edits + admin actions ride on the
    # UI's session token (or any future operator-scoped client). Service
    # tokens are rejected.
    operator_only = [Depends(require_operator)]
    app.include_router(config_routes.router, dependencies=operator_only)
    app.include_router(admin_routes.router, dependencies=operator_only)

    # Mixed surfaces: chat and conversation reads are reachable from
    # both the UI (operator token) and peer components (service tokens
    # — e.g. connector → orchestrator when a Discord message comes in).
    authorized = [Depends(require_authorized)]
    app.include_router(conversations_routes.router, dependencies=authorized)
    app.include_router(chat_routes.router, dependencies=authorized)

    return app
