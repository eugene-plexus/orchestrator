"""Entrypoint: `python -m eugene_plexus_orchestrator`."""

from __future__ import annotations

import uvicorn

from .app import create_app
from .config import ConfigStore
from .settings import load_settings


def main() -> None:
    settings = load_settings()

    bootstrap_store = ConfigStore(settings.config_file)
    bootstrap_store.load()
    port = int(bootstrap_store.get("port") or 8080)
    log_level = str(bootstrap_store.get("logLevel") or "INFO").lower()

    app = create_app(settings)
    uvicorn.run(app, host=settings.bind_host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
