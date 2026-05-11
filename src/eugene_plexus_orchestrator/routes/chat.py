"""POST /v1/chat (real) and POST /v1/chat/stream (still 501)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, HTTPException, Request, status

from .._generated.models import (
    ChatRequest,
    ChatResponse,
    Constitution,
    MemoryEntry,
    Message,
    Person,
    Problem,
    RelationshipSummary,
    Role,
    SelfModelEntry,
)
from ..bicameral.loop import run_bicameral_loop
from ..bicameral.nt import neutral_state
from ..config import ConfigStore
from ..hemisphere_client import HemisphereClient, HemisphereDriverError
from ..identity import IdentityClient
from ..memory import NIL_PERSON_ID, MemoryClient

router = APIRouter(tags=["chat"])

log = logging.getLogger(__name__)


def _build_history(history: list[Message], user_message: str) -> list[Message]:
    """Conversation history to send to hemispheres for this turn.

    Returns user / assistant turns plus the new user message — no system
    prompt (the bicameral loop prepends per-driver system prompts).
    Hemisphere-tagged intermediate messages from prior turns are
    observability artifacts and don't belong in subsequent prompts.
    """
    out: list[Message] = []
    for msg in history:
        if msg.role in (Role.user, Role.assistant):
            out.append(msg)
    out.append(Message(role=Role.user, content=user_message))
    return out


async def _resolve_operator_person_id(identity: IdentityClient) -> UUID | None:
    """Look up the operator's personId from the identity component.

    UI chat calls don't supply `personId`; the orchestrator resolves it
    on demand. Returns None when no operator person exists yet (e.g. an
    install whose first-run wizard hasn't run `ensure_operator`).
    """
    persons = await identity.list_persons()
    for p in persons:
        if p.isOperator:
            return p.personId
    return None


def _build_memory_entry(
    *,
    conversation_id: UUID,
    person_id: UUID,
    message: Message,
    nt_snapshot: Any = None,
    hemisphere_attribution: str | None = None,
) -> MemoryEntry:
    """Wrap a `Message` in a `MemoryEntry` with full v0.2 metadata.

    Callers always know `conversation_id` (URL) and `person_id` (body
    or operator fallback). The orchestrator is the source of truth for
    NT snapshot and hemisphere attribution — the memory component
    stores both as opaque blobs.
    """
    return MemoryEntry(
        entryId=uuid4(),
        personId=person_id,
        conversationId=conversation_id,
        role=message.role,
        content=message.content,
        timestamp=message.timestamp or datetime.now(UTC),
        ntStateSnapshot=nt_snapshot,
        hemisphereAttribution=hemisphere_attribution,
    )


def _render_recent_turns(entries: list[MemoryEntry]) -> str:
    """Render recent memory turns as a compact prompt section.

    The orchestrator pulls these for the speaker's `personId` and
    injects them into per-hemisphere prompts so Eugene has concrete
    context (not just identity's relationship summary). Each entry
    surfaces as one line; the bicameral loop already supplies the full
    user/assistant turns via `history`, so we omit any entry whose
    timestamp is in the active conversation — those are already in
    scope.
    """
    if not entries:
        return ""
    # Newest first → reverse for readability (oldest mention first).
    lines = ["Recent turns with this person (oldest first):"]
    for e in reversed(entries):
        label = "you" if e.role == Role.assistant else "they"
        # Truncate per-entry to keep the prompt bounded.
        content = e.content if len(e.content) <= 280 else e.content[:280] + "…"
        lines.append(f"- {label}: {content}")
    return "\n".join(lines)


def _hemisphere_preamble(this_driver: str, twin_driver: str, *, position: str) -> str:
    """Per-hemisphere preamble identifying which side this is.

    `position` is "left" or "right" — matches the operator's tab order
    in the UI, not an anatomical claim. The twin attribution makes the
    cross-vendor bicameral commitment visible to the model: each
    hemisphere knows the OTHER one is running on a different backend
    and will respond in parallel without seeing this side's output.
    """
    twin_position = "right" if position == "left" else "left"
    return (
        f"You are Eugene's {position} hemisphere, running on backend {this_driver!r}. "
        f"Your twin (Eugene's {twin_position} hemisphere) runs on backend "
        f"{twin_driver!r} and will respond in parallel — you don't see their "
        "output while you think. The orchestrator (corpus callosum) blends "
        "both responses into Eugene's final reply."
    )


def _render_constitution(c: Constitution) -> str:
    parts: list[str] = []
    parts.append(f"Your name: {c.name}.")
    if c.pronouns:
        parts.append(f"Pronouns: {c.pronouns}.")
    if c.coreValues:
        parts.append("Core values: " + "; ".join(c.coreValues) + ".")
    if c.freeText:
        parts.append(c.freeText.strip())
    return "\n".join(parts)


def _render_self_model(entries: list[SelfModelEntry]) -> str:
    if not entries:
        return ""
    lines = ["What you've come to notice about yourself:"]
    for e in entries:
        lines.append(f"- [{e.topic}] {e.content}")
    return "\n".join(lines)


def _render_relationship(summary: RelationshipSummary, person: Person | None) -> str:
    """Render relationship context for the speaker.

    Prefer the synthesized `summary` when available; v0.2's default is
    `recentTurns`-only (no summary) so fall back to a recent-turn count
    and the operator's free-form `relationshipNote` if present.
    """
    parts: list[str] = []
    if person is not None:
        intro = f"You are talking to {person.displayName}"
        if person.isOperator:
            intro += " (your operator — the person who set up your install)"
        intro += "."
        parts.append(intro)
        if person.relationshipNote:
            parts.append(f"Operator note about them: {person.relationshipNote}")
    if summary.summary:
        parts.append(summary.summary)
    elif summary.turnCount and summary.turnCount > 0:
        parts.append(
            f"You've shared {summary.turnCount} prior turn(s) with this person."
        )
    return "\n".join(parts)


async def _build_per_driver_system_prompts(
    *,
    drivers: list[HemisphereClient],
    identity: IdentityClient | None,
    person_id: UUID | None,
    operator_person_id: UUID | None,
    user_message: str,
    operator_override: str | None,
    fallback_default: str,
    recent_turns: list[MemoryEntry] | None = None,
) -> dict[str, str]:
    """Assemble per-hemisphere system prompts.

    When `identity` is configured, each driver gets a preamble + the
    full identity stack (constitution + self-model + relationship). When
    identity is not configured (or fails partially), every driver gets
    the same shared system prompt — v0.1's behavior — preceded only by
    the hemisphere preamble.

    `operator_override` is `ChatRequest.systemPrompt`. When provided, it
    replaces the identity-assembled persona body; the per-hemisphere
    preamble is still added on top.
    """
    if len(drivers) != 2:
        # The bicameral loop enforces this too — short-circuit cleanly
        # so the preamble generator below doesn't have to handle N!=2.
        raise ValueError(
            f"per-driver prompt assembly requires exactly two drivers; got {len(drivers)}"
        )

    left, right = drivers[0], drivers[1]
    preamble_left = _hemisphere_preamble(left.name, right.name, position="left")
    preamble_right = _hemisphere_preamble(right.name, left.name, position="right")

    persona_body: str
    if operator_override:
        persona_body = operator_override
    elif identity is not None:
        # Resolve the speaker's personId: explicit body.personId wins;
        # otherwise default to the operator (UI chat calls without a
        # personId are by-design operator turns).
        effective_person_id = person_id or operator_person_id
        constitution = await identity.get_constitution()
        # v0.2 has no topic-detection yet; pass the raw user message as
        # the topic hint. The identity component does a starts-with /
        # exact-match against `topic` and falls back to recency, so the
        # full-message-as-topic produces near-recency behavior here —
        # good enough until v0.3 adds the topic-shift detector.
        self_model_task = identity.query_self_model(
            topic=None,
            person_id=effective_person_id,
            limit=5,
        )
        if effective_person_id is not None:
            relationship_task = identity.get_relationship(effective_person_id)
            persons_task = identity.list_persons()
        else:
            relationship_task = None
            persons_task = None
        self_model_entries = await self_model_task
        relationship = await relationship_task if relationship_task is not None else None
        persons = await persons_task if persons_task is not None else []
        person_record = next(
            (p for p in persons if p.personId == effective_person_id), None
        ) if effective_person_id is not None else None

        sections: list[str] = [_render_constitution(constitution)]
        sm_text = _render_self_model(self_model_entries)
        if sm_text:
            sections.append(sm_text)
        if relationship is not None:
            rel_text = _render_relationship(relationship, person_record)
            if rel_text:
                sections.append(rel_text)
        elif person_record is not None:
            # Person known but no relationship summary (e.g. brand new
            # person). Still surface name + operator note so Eugene
            # doesn't speak to them generically.
            intro = f"You are talking to {person_record.displayName}"
            if person_record.isOperator:
                intro += " (your operator — the person who set up your install)"
            intro += "."
            if person_record.relationshipNote:
                intro += f"\nOperator note about them: {person_record.relationshipNote}"
            sections.append(intro)
        persona_body = "\n\n".join(s for s in sections if s)
    else:
        # v0.1 fallback path — no identity component configured.
        persona_body = fallback_default

    # Append recent-turns context (the memory-side enrichment, separate
    # from identity's constitution/self-model). Skipped when empty or
    # when the operator override is suppressing the persona body —
    # operator override is an explicit "don't enrich" signal.
    if recent_turns and not operator_override:
        recent_section = _render_recent_turns(recent_turns)
        if recent_section:
            persona_body = f"{persona_body}\n\n{recent_section}".strip()

    # Reference user_message to keep mypy happy even though we don't (yet)
    # synthesize per-turn topic queries from it. v0.3's topic-shift
    # detector will consume this.
    _ = user_message

    return {
        left.name: f"{preamble_left}\n\n{persona_body}".strip(),
        right.name: f"{preamble_right}\n\n{persona_body}".strip(),
    }


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    if getattr(request.app.state, "safe_mode", False):
        # Per the safe-mode contract: orchestrator stays reachable for
        # config edits but its primary endpoint returns 503 until the
        # operator fixes config and restarts without the env var.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=Problem(
                type="https://github.com/eugene-plexus/orchestrator#safe-mode",
                title="Orchestrator is in safe mode",
                status=503,
                detail=(
                    "The orchestrator was started with "
                    "EUGENE_PLEXUS_ORCH_SAFE_MODE=1 and is running on "
                    "built-in defaults. Fix the on-disk config via "
                    "PATCH /v1/config, then restart without the env var."
                ),
                component="orchestrator",
            ).model_dump(exclude_none=True),
        )

    store: ConfigStore = request.app.state.config_store
    memory: MemoryClient = request.app.state.memory
    drivers: list[HemisphereClient] = request.app.state.drivers
    identity: IdentityClient | None = getattr(request.app.state, "identity", None)

    # Resolve speaker before any memory writes so we can stamp every
    # MemoryEntry with the right personId. body.personId wins; otherwise
    # fall back to the operator's personId from identity. UI chat calls
    # have no personId by design — they're always operator turns.
    operator_person_id: UUID | None = None
    if identity is not None and body.personId is None:
        try:
            operator_person_id = await _resolve_operator_person_id(identity)
        except httpx.HTTPError as e:
            log.warning(
                "identity service unreachable while resolving operator: %s "
                "(falling back to no-person-context path)",
                e,
            )
            # Don't fail the chat turn just because identity is down —
            # degrade to no-person-context and continue.
            operator_person_id = None
    effective_person_id = body.personId or operator_person_id or NIL_PERSON_ID

    nt_at_start = neutral_state()

    try:
        if body.conversationId is not None:
            existing = await memory.get(body.conversationId)
            if existing is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=Problem(
                        type="https://github.com/eugene-plexus/orchestrator#conversation-not-found",
                        title="Conversation not found",
                        status=404,
                        detail=f"No conversation with id {body.conversationId}.",
                        component="orchestrator",
                    ).model_dump(exclude_none=True),
                )
            conversation_id = body.conversationId
            history = list(existing.messages)
        else:
            conversation_id = await memory.create()
            history = []

        user_message = Message(role=Role.user, content=body.message)
        await memory.append_entry(
            conversation_id,
            _build_memory_entry(
                conversation_id=conversation_id,
                person_id=effective_person_id,
                message=user_message,
                nt_snapshot=nt_at_start,
            ),
        )
    except httpx.HTTPError as e:
        log.warning("memory service unreachable: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=Problem(
                type="https://github.com/eugene-plexus/orchestrator#memory-error",
                title="Memory service error",
                status=502,
                detail=f"Memory service at {request.app.state.memory_url} is unreachable: {e}",
                component="orchestrator",
            ).model_dump(exclude_none=True),
        ) from e

    # Pull recent turns with this person from memory — feeds the
    # relationship-context section of the per-hemisphere prompts. Skip
    # for NIL_PERSON_ID (we'd just get unrelated NIL-bucket entries).
    # Failure here degrades silently: prompts still get built without
    # the recent-turns section.
    recent_with_person: list[MemoryEntry] = []
    if effective_person_id != NIL_PERSON_ID:
        try:
            recent_with_person = await memory.person_recent(
                effective_person_id,
                limit=int(store.get("personRecentLimit") or 10),
                # Exclude the active conversation — its turns are already
                # in `history` and don't need to be replayed as ambient
                # context.
            )
            recent_with_person = [
                e for e in recent_with_person if e.conversationId != conversation_id
            ]
        except httpx.HTTPError as e:
            log.warning(
                "memory service unreachable while fetching person_recent: %s "
                "(continuing without recent-turns context)",
                e,
            )

    try:
        system_prompts = await _build_per_driver_system_prompts(
            drivers=drivers,
            identity=identity,
            person_id=body.personId,
            operator_person_id=operator_person_id,
            user_message=body.message,
            operator_override=body.systemPrompt,
            fallback_default=str(store.get("defaultSystemPrompt") or ""),
            recent_turns=recent_with_person,
        )
    except httpx.HTTPError as e:
        log.warning(
            "identity service unreachable while assembling system prompts: %s "
            "(falling back to defaultSystemPrompt)",
            e,
        )
        # Identity is down mid-assembly. Use defaultSystemPrompt for both
        # hemispheres (still with the per-hemisphere preamble) and
        # continue — chat works even with identity offline.
        left, right = drivers[0], drivers[1]
        fallback = body.systemPrompt or str(store.get("defaultSystemPrompt") or "")
        left_preamble = _hemisphere_preamble(left.name, right.name, position="left")
        right_preamble = _hemisphere_preamble(right.name, left.name, position="right")
        system_prompts = {
            left.name: f"{left_preamble}\n\n{fallback}".strip(),
            right.name: f"{right_preamble}\n\n{fallback}".strip(),
        }

    history_for_drivers = _build_history(history, body.message)

    nt_at_start = neutral_state()
    max_passes = int(body.maxPasses or store.get("defaultMaxPasses") or 3)
    agreement_threshold = float(store.get("agreementThreshold") or 0.5)
    # LLM-output-affecting params are owned by the orchestrator. v0.1 reads
    # static defaults from config; v0.2+ will derive them from NT state.
    temperature_cfg = store.get("defaultTemperature")
    temperature = float(temperature_cfg) if temperature_cfg is not None else None
    max_tokens_cfg = store.get("defaultMaxTokens")
    max_tokens = int(max_tokens_cfg) if max_tokens_cfg is not None else None

    try:
        outcome = await run_bicameral_loop(
            history=history_for_drivers,
            system_prompts=system_prompts,
            drivers=drivers,
            nt_state=nt_at_start,
            max_passes=max_passes,
            agreement_threshold=agreement_threshold,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except HemisphereDriverError as e:
        # The driver returned a structured error — surface its actual
        # `Problem` body to the user instead of the generic httpx text.
        # Most useful failure mode: model-specific upstream errors
        # (OpenAI rejecting temperature, missing CLI binary, etc.) come
        # through readable rather than as an opaque "502 Bad Gateway".
        log.warning(
            "hemisphere %r returned %d: %s",
            e.driver_name,
            e.status_code,
            e.problem.detail if e.problem else e.raw_body[:200],
        )
        upstream = e.problem
        upstream_title = upstream.title if upstream else "Hemisphere driver error"
        upstream_detail = upstream.detail if upstream else (e.raw_body[:500] or "(no body)")
        upstream_component = (
            upstream.component if upstream else f"hemisphere-driver:{e.driver_name}"
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=Problem(
                type="https://github.com/eugene-plexus/orchestrator#hemisphere-error",
                title=f"Hemisphere {e.driver_name!r} failed: {upstream_title}",
                status=502,
                detail=(
                    f"{upstream_detail} "
                    f"(driver={e.driver_name}, url={e.driver_url}, "
                    f"upstream-status={e.status_code}, "
                    f"upstream-component={upstream_component})"
                ),
                component="orchestrator",
            ).model_dump(exclude_none=True),
        ) from e
    except httpx.HTTPError as e:
        # Network-level failure: connection refused, timeout, DNS, etc.
        # No driver-side body to extract — fall back to the httpx string.
        log.warning("bicameral loop failed at HTTP layer: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=Problem(
                type="https://github.com/eugene-plexus/orchestrator#hemisphere-error",
                title="Hemisphere driver unreachable",
                status=502,
                detail=str(e),
                component="orchestrator",
            ).model_dump(exclude_none=True),
        ) from e

    try:
        await memory.append_entry(
            conversation_id,
            _build_memory_entry(
                conversation_id=conversation_id,
                person_id=effective_person_id,
                message=outcome.final_message,
                nt_snapshot=nt_at_start,
                hemisphere_attribution="blended",
            ),
        )
    except httpx.HTTPError as e:
        log.warning("memory service unreachable while persisting reply: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=Problem(
                type="https://github.com/eugene-plexus/orchestrator#memory-error",
                title="Memory service error",
                status=502,
                detail=f"Memory service at {request.app.state.memory_url} is unreachable: {e}",
                component="orchestrator",
            ).model_dump(exclude_none=True),
        ) from e

    return ChatResponse(
        conversationId=conversation_id,
        message=outcome.final_message,
        passes=outcome.passes,
        ntStateAtStart=nt_at_start,
        ntStateAtEnd=nt_at_start,
        requestId=body.requestId,
    )


@router.post("/v1/chat/stream")
async def chat_stream(body: ChatRequest) -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=Problem(
            type="https://github.com/eugene-plexus/orchestrator#not-implemented",
            title="Not Implemented",
            status=501,
            detail=(
                "POST /v1/chat/stream is not yet wired up. Will land alongside "
                "the UI consumer and hemisphere-driver streaming in a v0.1 "
                "follow-up."
            ),
            component="orchestrator",
        ).model_dump(exclude_none=True),
    )
