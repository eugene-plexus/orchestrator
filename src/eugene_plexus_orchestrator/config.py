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
    "topology": "Drivers",
    "memory": "Memory Service",
    "identity": "Identity Service",
    "network": "Network",
    "logging": "Logging",
    "bicameral": "Bicameral Loop",
    "generation": "Generation Defaults (NT-modulated in v0.2+)",
    "persona": "Persona",
}

# Default driver topology: the canonical bicameral pair on local ports.
# Operators rename / re-URL these via PATCH /v1/config; v0.2+ adds N drivers
# with backup semantics on top of the same shape.
DEFAULT_DRIVERS: list[dict[str, str]] = [
    {"name": "left", "url": "http://127.0.0.1:8081"},
    {"name": "right", "url": "http://127.0.0.1:8082"},
]

DEFAULT_SYSTEM_PROMPT = (
    "You are Eugene, a synthetic consciousness scaffolded by the Eugene Plexus "
    "framework. Two of you exist in superposition during each turn — a left "
    "hemisphere and a right hemisphere. Your responses will be reconciled by a "
    "corpus-callosum process. Be thoughtful, concise, and honest."
)

FIELDS: list[ConfigField] = [
    ConfigField(
        key="drivers",
        label="Drivers",
        description=(
            "The list of LLM drivers (\"hemispheres\") the orchestrator "
            "talks to on every chat turn. Each entry has a `name` "
            "(free-form label, e.g. \"left\" / \"right\" / \"claude\" / "
            "\"local-llama\" — appears on the UI's tabs and beside that "
            "driver's outputs) and a `url` (HTTP base of a running "
            "hemisphere-driver, e.g. `http://127.0.0.1:8081`). v0.1's "
            "bicameral loop requires exactly two entries; v0.2+ will "
            "generalize to N with backup/failover. Use the per-row "
            "`Test` button to verify a URL is reachable before saving."
        ),
        category="topology",
        valueType=ConfigValueType.driver_list,
        default=DEFAULT_DRIVERS,
        required=True,
        requiresRestart=True,
    ),
    ConfigField(
        key="memoryUrl",
        label="Memory service URL",
        description=(
            "HTTP base of the running `eugene-plexus/memory` service — "
            "where conversation history is stored and retrieved. v0.1 "
            "ships an in-process memory backend on port 8083 by default."
        ),
        category="memory",
        valueType=ConfigValueType.url,
        default="http://127.0.0.1:8083",
        required=True,
        requiresRestart=True,
    ),
    ConfigField(
        key="identityUrl",
        label="Identity service URL",
        description=(
            "HTTP base of the running `eugene-plexus/identity` service — "
            "Eugene's constitution + self-model + person registry. When "
            "set, the orchestrator pulls constitution + relevant "
            "self-model entries + the speaker's relationship summary on "
            "every chat turn and assembles per-hemisphere system prompts "
            "from them. Leave unset to fall back to v0.1's single "
            "shared-system-prompt path (uses `defaultSystemPrompt`)."
        ),
        category="identity",
        valueType=ConfigValueType.url,
        default=None,
        required=False,
        requiresRestart=True,
    ),
    # The orchestrator's bind port used to live here. It moved out:
    # ports are owned by the watchdog topology now and passed to spawned
    # children via EUGENE_PLEXUS_ORCH_BIND_PORT. One source of truth
    # avoids the OpenClaw-style "config says 8080 but watchdog spawned
    # at 8090, nobody can reach the orchestrator" trap.
    ConfigField(
        key="logLevel",
        label="Log level",
        description=(
            "How chatty the orchestrator's terminal output is. `DEBUG` "
            "prints every bicameral pass and per-driver dispatch "
            "(useful for debugging); `INFO` is the normal level; "
            "`WARNING` and `ERROR` go progressively quieter."
        ),
        category="logging",
        valueType=ConfigValueType.enum,
        default="INFO",
        enumValues=["DEBUG", "INFO", "WARNING", "ERROR"],
        requiresRestart=True,
    ),
    ConfigField(
        key="defaultMaxPasses",
        label="Default max passes",
        description=(
            "Hard cap on how many times the orchestrator will re-prompt "
            "the hemispheres before giving up and returning a blended "
            "response anyway. Each \"pass\" sends the conversation to "
            "every driver in parallel and scores how much they agree; "
            "if they disagree, another pass runs. Higher values give "
            "the system more chances to converge on a unified answer at "
            "the cost of latency and tokens. The chat request can "
            "override this per-call."
        ),
        category="bicameral",
        valueType=ConfigValueType.integer,
        default=3,
        minimum=1,
        maximum=10,
    ),
    ConfigField(
        key="agreementThreshold",
        label="Agreement threshold",
        description=(
            "How much overlap two driver responses need before the "
            "orchestrator considers them \"in agreement\" and stops "
            "looping. Computed as Jaccard similarity over the unique "
            "words in each response — 0.0 is no overlap at all, 1.0 is "
            "identical text. Lower values are more permissive (loop "
            "ends sooner); higher values demand near-identical answers."
        ),
        category="bicameral",
        valueType=ConfigValueType.number,
        default=0.5,
        minimum=0.0,
        maximum=1.0,
    ),
    ConfigField(
        key="defaultSystemPrompt",
        label="Default system prompt",
        description=(
            "Persona/instructions sent to every driver as a `system` "
            "message at the start of each chat turn. Used unless the "
            "incoming chat request supplies its own `systemPrompt`. "
            "This is where Eugene's voice gets established — \"You are "
            "Eugene…\" — and where you'd add any global instructions."
        ),
        category="persona",
        valueType=ConfigValueType.string,
        default=DEFAULT_SYSTEM_PROMPT,
    ),
    ConfigField(
        key="defaultTemperature",
        label="Default temperature",
        description=(
            "Sampling randomness sent to every driver on every call. "
            "0 = deterministic / always picks the most likely next "
            "token. 1 = the model's own default randomness. Higher "
            "values get more creative / more varied / more unhinged. "
            "v0.1 sends a static value; v0.2+ will modulate it from "
            "neurotransmitter state per-pass per-driver."
        ),
        category="generation",
        valueType=ConfigValueType.number,
        default=0.7,
        minimum=0.0,
        maximum=2.0,
    ),
    ConfigField(
        key="defaultMaxTokens",
        label="Default max output tokens",
        description=(
            "Cap on how long a single driver response can be (roughly "
            "0.75 words per token). 2048 is enough for ~1,500 words of "
            "output. Bump it for long-form work; keep it low for "
            "snappier chat. v0.1 placeholder for the future NT system."
        ),
        category="generation",
        valueType=ConfigValueType.integer,
        default=2048,
        minimum=1,
    ),
    ConfigField(
        key="requestTimeoutSeconds",
        label="Driver request timeout",
        description=(
            "How long the orchestrator waits on one driver's response "
            "before giving up. Counts from the start of the HTTP "
            "request to the driver, NOT the end of the bicameral loop. "
            "Bump this if your slowest model needs more time per pass."
        ),
        category="bicameral",
        valueType=ConfigValueType.duration,
        default=180,
        minimum=5,
        maximum=900,
        requiresRestart=True,
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

    if vt == ConfigValueType.driver_list:
        if not isinstance(value, list):
            return f"expected list of {{name, url}} entries, got {type(value).__name__}"
        if not value:
            return "drivers list must not be empty"
        seen_names: set[str] = set()
        for i, entry in enumerate(value):
            if not isinstance(entry, dict):
                return f"entry {i}: expected object, got {type(entry).__name__}"
            name = entry.get("name")
            url = entry.get("url")
            if not isinstance(name, str) or not name.strip():
                return f"entry {i}: `name` must be a non-empty string"
            if not isinstance(url, str) or not url.strip():
                return f"entry {i}: `url` must be a non-empty string"
            if name in seen_names:
                return f"entry {i}: duplicate driver name {name!r}"
            seen_names.add(name)
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
