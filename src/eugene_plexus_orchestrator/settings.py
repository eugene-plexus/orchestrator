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


def load_settings() -> Settings:
    return Settings()
