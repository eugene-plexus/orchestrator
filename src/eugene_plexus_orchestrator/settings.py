"""Startup-time settings, sourced from environment variables.

Distinct from the runtime *config* (see `config.py`), which is editable via
`PATCH /v1/config`. These settings only control bootstrap.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EUGENE_PLEXUS_ORCH_",
        env_file=None,
        case_sensitive=False,
    )

    config_file: Path = Path("config.yaml")
    """Where the runtime config is persisted. PATCH /v1/config writes here."""

    bind_host: str = "127.0.0.1"
    """Network interface to bind. Override to 0.0.0.0 for tailnet exposure."""

    safe_mode: bool = False
    """If true, skip loading the persisted config file at startup and run on
    built-in defaults (no drivers configured, default memory URL). Set by
    the watchdog via EUGENE_PLEXUS_ORCH_SAFE_MODE=1 when a previous boot
    failed. PATCH /v1/config still writes to `config_file` normally so
    the operator's repair survives the next non-safe-mode boot. Per the
    safe-mode contract in specs/openapi/orchestrator.yaml."""

    auth_signing_key: str | None = None
    """Base64-encoded 32-byte HMAC signing key, supplied by the watchdog at
    spawn time (EUGENE_PLEXUS_ORCH_AUTH_SIGNING_KEY). When absent the
    orchestrator runs unauthenticated — dev / standalone path only;
    production via the watchdog always supplies this."""

    service_token: str | None = None
    """Long-lived service JWT for outbound calls to peer components
    (EUGENE_PLEXUS_ORCH_SERVICE_TOKEN). Required when `auth_signing_key`
    is set; the orchestrator presents this on every outbound httpx call."""

    master_key: str | None = None
    """Base64-encoded 32-byte secretbox key for at-rest decryption
    (EUGENE_PLEXUS_ORCH_MASTER_KEY). Populated only after the operator
    has logged in at the watchdog; absent during the configured-but-
    locked window. Reserved for Phase 6; Phase 3 does not consume it."""


def load_settings() -> Settings:
    return Settings()
