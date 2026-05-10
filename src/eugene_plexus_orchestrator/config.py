"""Runtime configuration: schema declaration + file-backed state + PATCH apply.

Implements the shared Eugene Plexus config protocol on the orchestrator
(`GET /v1/config/schema`, `GET /v1/config`, `PATCH /v1/config`). This is the
same code shape as in `hemisphere-driver/config.py` — the two have to agree
on protocol semantics so a single UI can edit both.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import yaml

from ._generated.models import (
    ConfigDocument,
    ConfigField,
    ConfigFieldError,
    ConfigSchema,
    ConfigUpdateRequest,
    ConfigUpdateResult,
    ConfigValueType,
)

REDACTED = "<redacted>"

CATEGORY_LABELS: dict[str, str] = {
    "hemispheres": "Hemisphere Pair",
    "memory": "Memory Service",
    "network": "Network",
    "logging": "Logging",
    "bicameral": "Bicameral Loop",
    "persona": "Persona",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are Eugene, a synthetic consciousness scaffolded by the Eugene Plexus "
    "framework. Two of you exist in superposition during each turn — a left "
    "hemisphere and a right hemisphere. Your responses will be reconciled by a "
    "corpus-callosum process. Be thoughtful, concise, and honest."
)

FIELDS: list[ConfigField] = [
    ConfigField(
        key="leftDriverUrl",
        label="Left Hemisphere URL",
        description="Base URL of the hemisphere-driver instance acting as the left hemisphere.",
        category="hemispheres",
        valueType=ConfigValueType.url,
        default="http://127.0.0.1:8081",
        required=True,
    ),
    ConfigField(
        key="rightDriverUrl",
        label="Right Hemisphere URL",
        description="Base URL of the hemisphere-driver instance acting as the right hemisphere.",
        category="hemispheres",
        valueType=ConfigValueType.url,
        default="http://127.0.0.1:8082",
        required=True,
    ),
    ConfigField(
        key="memoryUrl",
        label="Memory Service URL",
        description=(
            "Base URL of the eugene-plexus/memory service. The orchestrator "
            "delegates conversation persistence here. Restart required so "
            "the HTTP client picks up the new base URL."
        ),
        category="memory",
        valueType=ConfigValueType.url,
        default="http://127.0.0.1:8083",
        required=True,
        requiresRestart=True,
    ),
    ConfigField(
        key="port",
        label="HTTP Port",
        description="Port to listen on.",
        category="network",
        valueType=ConfigValueType.integer,
        default=8080,
        minimum=1,
        maximum=65535,
        requiresRestart=True,
    ),
    ConfigField(
        key="logLevel",
        label="Log Level",
        description="Logging verbosity.",
        category="logging",
        valueType=ConfigValueType.enum,
        default="INFO",
        enumValues=["DEBUG", "INFO", "WARNING", "ERROR"],
    ),
    ConfigField(
        key="defaultMaxPasses",
        label="Default Max Passes",
        description=(
            "Maximum bicameral passes per turn before the orchestrator forces "
            "termination regardless of hemisphere disagreement."
        ),
        category="bicameral",
        valueType=ConfigValueType.integer,
        default=3,
        minimum=1,
        maximum=10,
    ),
    ConfigField(
        key="agreementThreshold",
        label="Agreement Threshold",
        description=(
            "Word-set Jaccard similarity above which the corpus callosum "
            "decides the hemispheres agree and terminates the loop."
        ),
        category="bicameral",
        valueType=ConfigValueType.number,
        default=0.5,
        minimum=0.0,
        maximum=1.0,
    ),
    ConfigField(
        key="defaultSystemPrompt",
        label="Default System Prompt",
        description=(
            "System prompt used when the chat request doesn't supply its own. "
            "Establishes Eugene's persona to the underlying LLMs."
        ),
        category="persona",
        valueType=ConfigValueType.string,
        default=DEFAULT_SYSTEM_PROMPT,
    ),
    ConfigField(
        key="requestTimeoutSeconds",
        label="Driver Request Timeout",
        description="Maximum seconds to wait for one hemisphere-driver response.",
        category="bicameral",
        valueType=ConfigValueType.duration,
        default=180,
        minimum=5,
        maximum=900,
    ),
]

_FIELDS_BY_KEY: dict[str, ConfigField] = {f.key: f for f in FIELDS}


def as_schema() -> ConfigSchema:
    return ConfigSchema(
        component="orchestrator",
        fields=FIELDS,
        categories=CATEGORY_LABELS,
    )


def _defaults() -> dict[str, Any]:
    return {f.key: f.default for f in FIELDS if f.default is not None}


def _validate_value(field: ConfigField, value: Any) -> str | None:
    if value is None:
        return None

    vt = field.valueType

    if vt in (ConfigValueType.string, ConfigValueType.url, ConfigValueType.file_path):
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        if field.pattern is not None:
            import re

            if re.search(field.pattern, value) is None:
                return f"value does not match pattern {field.pattern!r}"
        return None

    if vt == ConfigValueType.secret:
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        if value == REDACTED:
            return "refusing to write the literal redacted value back"
        return None

    if vt == ConfigValueType.integer:
        if isinstance(value, bool) or not isinstance(value, int):
            return f"expected integer, got {type(value).__name__}"
        if field.minimum is not None and value < field.minimum:
            return f"must be >= {field.minimum}"
        if field.maximum is not None and value > field.maximum:
            return f"must be <= {field.maximum}"
        return None

    if vt in (ConfigValueType.number, ConfigValueType.duration):
        if isinstance(value, bool) or not isinstance(value, int | float):
            return f"expected number, got {type(value).__name__}"
        if field.minimum is not None and value < field.minimum:
            return f"must be >= {field.minimum}"
        if field.maximum is not None and value > field.maximum:
            return f"must be <= {field.maximum}"
        return None

    if vt == ConfigValueType.boolean:
        if not isinstance(value, bool):
            return f"expected boolean, got {type(value).__name__}"
        return None

    if vt == ConfigValueType.enum:
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        allowed = field.enumValues or []
        if value not in allowed:
            return f"must be one of {allowed}"
        return None

    return f"unsupported valueType: {vt}"


class ConfigStore:
    """File-backed config state. Thread-safe for the simple read/write pattern."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._values: dict[str, Any] = _defaults()
        self._pending_restart: set[str] = set()

    def load(self) -> None:
        with self._lock:
            if self._path.exists():
                raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
                if not isinstance(raw, dict):
                    raise ValueError(f"config file {self._path} must be a YAML mapping at the root")
                merged = _defaults()
                for k, v in raw.items():
                    if k in _FIELDS_BY_KEY:
                        merged[k] = v
                self._values = merged
            else:
                self._values = _defaults()
                self._write_locked()

    def as_document(self) -> ConfigDocument:
        with self._lock:
            out: dict[str, Any] = {}
            for key, value in self._values.items():
                field = _FIELDS_BY_KEY.get(key)
                if field is not None and field.sensitive and value is not None:
                    out[key] = REDACTED
                else:
                    out[key] = value
            return ConfigDocument.model_validate(out)

    def apply_patch(self, request: ConfigUpdateRequest) -> ConfigUpdateResult:
        applied: list[str] = []
        rejected: list[ConfigFieldError] = []
        pending_restart: list[str] = []

        patch: dict[str, Any] = request.model_dump()

        with self._lock:
            for key, new_value in patch.items():
                field = _FIELDS_BY_KEY.get(key)
                if field is None:
                    rejected.append(ConfigFieldError(key=key, message="unknown field"))
                    continue

                err = _validate_value(field, new_value)
                if err is not None:
                    rejected.append(ConfigFieldError(key=key, message=err))
                    continue

                if new_value is None and field.default is not None:
                    self._values[key] = field.default
                else:
                    self._values[key] = new_value

                applied.append(key)
                if field.requiresRestart:
                    self._pending_restart.add(key)
                    pending_restart.append(key)

            if applied:
                self._write_locked()

            requires_restart = bool(self._pending_restart)
            return ConfigUpdateResult(
                applied=applied,
                rejected=rejected,
                requiresRestart=requires_restart,
                pendingRestart=sorted(self._pending_restart),
            )

    def get(self, key: str) -> Any:
        with self._lock:
            return self._values.get(key)

    def _write_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self._values, f, sort_keys=True, default_flow_style=False)
