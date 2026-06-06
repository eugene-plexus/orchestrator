"""The consciousness loop — Eugene's single, continuous cognitive task.

One Eugene = one consciousness = one workspace = one attention. This loop
is the *only* thing that thinks: HTTP handlers inject `AfferentEvent`s
onto its queue (`POST /v1/events`) and subscribe to its observability
stream (`GET /v1/stream/consciousness`). Because exactly one task mutates
cognitive state, there is no locking anywhere in the cognition path.

**This first increment (M2 slice 2) is behavior-preserving-ish:** on a
`message` event the loop reproduces the v0.2 turn — recall → per-driver
prompts → bicameral deliberation → voice pass → NT tick → persist — but
restructured as fire-and-forget with the reply leaving asynchronously as
an `EfferentSpeechAct`, and every step published to the consciousness
stream. The action gate is DETERMINISTIC here (a message that produced a
deliberation is spoken). The real NT-valence gate, adenosine/sleep,
plateau-stop, presence, and multi-focus salience switching layer on in
later increments — see `docs/design/m1-continuous-runtime.md` in `specs`.

Errors NEVER crash the loop: there is no caller waiting on an HTTP
response, so a failed recall / deliberation / voice pass is logged (and,
where useful, surfaced on the stream) and the loop moves on.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime

import httpx
from fastapi import FastAPI

from .._generated.models import (
    AfferentEvent,
    EfferentSpeechAct,
    GateDecision,
    Message,
    NTState,
    Role,
)
from ..bicameral.loop import run_bicameral_loop
from ..bicameral.nt import modulated_temperature
from ..bicameral.voice import run_voice_pass
from ..hemisphere_client import HemisphereClient, HemisphereDriverError
from ..memory import NIL_PERSON_ID, MemoryClient
from ..tools import (
    TOOL_MEMORY_APPEND_ENTRY,
    TOOL_MEMORY_PERSON_RECENT,
    TOOL_NT_OBSERVE,
    ToolContext,
    ToolRunner,
    begin_tool_trace,
    new_call,
)
from . import turn
from .stream import ConsciousnessBroker
from .workspace import Workspace

log = logging.getLogger(__name__)

# Bound on pending afferent events. Single-attention means the loop drains
# one event per multi-second turn; an unbounded queue lets a faster-than-
# realtime producer grow memory without limit (and makes Eugene act on a
# stale flood). When full, POST /v1/events 503s — lossy-not-a-queue, the
# same principle the observability stream applies.
_EVENT_QUEUE_MAX = 256


def _enum_value(value: object) -> object:
    """Normalize a generated enum field to its underlying value.

    datamodel-code-generator emits `Enum` members for closed string enums;
    `--collapse-root-models` can also leave plain strings. Compare on the
    value either way.
    """
    return getattr(value, "value", value)


class ConsciousnessLoop:
    """Eugene's single continuous cognitive task.

    Constructed in the FastAPI lifespan after `app.state` is built, then
    `start()`ed. It reads its dependencies (drivers, memory, identity,
    scorer, tool_runner, config, NT) off `app.state` at handling time so
    it always sees current state (NT mutates turn to turn).
    """

    def __init__(self, app: FastAPI, broker: ConsciousnessBroker) -> None:
        self._app = app
        self._broker = broker
        self._queue: asyncio.Queue[AfferentEvent] = asyncio.Queue(maxsize=_EVENT_QUEUE_MAX)
        self._workspace = Workspace()
        self._running = False
        self._task: asyncio.Task[None] | None = None

    # -- lifecycle -----------------------------------------------------

    @property
    def queue(self) -> asyncio.Queue[AfferentEvent]:
        return self._queue

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    def submit(self, event: AfferentEvent) -> bool:
        """Enqueue an afferent event (called by `POST /v1/events`).

        Returns False when the queue is full — the loop can't keep up, so
        the route sheds the event (503) rather than growing memory with
        work it provably can't drain in time.
        """
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            return False

    def start(self) -> None:
        if self._task is None:
            self._running = True
            self._task = asyncio.create_task(self.run(), name="consciousness-loop")
            self._task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        """Clear the task handle and shout if cognition died unexpectedly.

        A dead loop that silently keeps accepting events (the queue fills,
        nothing drains it) is the worst failure mode; surface it loudly and
        clear the handle so a future `start()` can recreate the loop.
        """
        self._task = None
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("consciousness loop DIED: %r — cognition has stopped", exc)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def run(self) -> None:
        """The loop. Blocks on the queue; processes one event at a time.

        Single-attention is literal: one event is fully handled before the
        next is dequeued. Idle (no events) is a cheap block here; the
        seeking/mind-wandering idle floor + adenosine wake/sleep arrive in
        a later increment.
        """
        log.info("consciousness loop started")
        try:
            while self._running:
                event = await self._queue.get()
                try:
                    await self._handle(event)
                except Exception:
                    # The loop must never die on a single bad event.
                    log.exception("consciousness loop: unhandled error processing an event")
        except asyncio.CancelledError:
            log.info("consciousness loop cancelled")
            raise

    # -- dispatch ------------------------------------------------------

    async def _handle(self, event: AfferentEvent) -> None:
        kind = _enum_value(event.kind)
        if kind == "message":
            await self._handle_message(event)
        elif kind == "presence":
            # Presence (afferent occupancy) lands in M3 — NT-modulated
            # lossy salience + initiation. For now, observed and dropped.
            log.debug("presence event %s observed (M3 not yet wired)", event.eventId)
        else:
            log.warning("unknown afferent event kind %r — ignoring", kind)

    # -- the message turn (port of the v0.2 chat handler) --------------

    async def _handle_message(self, event: AfferentEvent) -> None:
        msg = event.message
        if msg is None:
            log.warning("message event %s has no message payload — dropping", event.eventId)
            return

        app = self._app
        if getattr(app.state, "safe_mode", False):
            log.warning("loop received a message in safe mode — nothing to think with; dropping")
            return
        drivers: list[HemisphereClient] = app.state.drivers
        if len(drivers) < 2:
            log.warning(
                "loop received a message but %d driver slot(s) resolved (need 2) — dropping",
                len(drivers),
            )
            return

        store = app.state.config_store
        memory: MemoryClient = app.state.memory
        tool_runner: ToolRunner = app.state.tool_runner
        scorer = app.state.scorer

        # Per-event tool trace. The loop is a single task that handles one
        # event fully before the next, and begin_tool_trace() re-sets the
        # task-local trace at the top of every turn — so records never bleed
        # across turns. (M3 note: any future non-message handler that runs
        # tools on this task must open its own begin_tool_trace scope.)
        trace = begin_tool_trace()
        # msg.channelContext (ambient platform messages preceding a mention)
        # rides on IncomingMessage but is not yet surfaced to hemispheres —
        # deferred like presence (M3), not dropped by oversight.

        # Focus shifts to this conversation. (Salience-driven switching
        # across multiple foci is a later increment; here every message
        # captures attention.)
        new_focus = str(msg.conversationId) if msg.conversationId else None
        if new_focus != self._workspace.focus:
            self._publish("focus_switch", {"from": self._workspace.focus, "to": new_focus})
            self._workspace.focus = new_focus

        nt_at_start: NTState = app.state.nt_state

        # Resolve the speaker. The connector always supplies a real
        # personId; a UI turn may post NIL_PERSON_ID to mean "this is the
        # operator" — resolve it from identity then.
        identity = getattr(app.state, "identity", None)
        operator_person_id = None
        if identity is not None and msg.personId == NIL_PERSON_ID:
            try:
                operator_person_id = await turn.resolve_operator_person_id(tool_runner)
            except httpx.HTTPError as e:
                log.warning(
                    "identity unreachable resolving operator: %s (degrading to no-person-context)",
                    e,
                )
        effective_person_id = (
            msg.personId if msg.personId != NIL_PERSON_ID else (operator_person_id or NIL_PERSON_ID)
        )

        # Load history + persist the incoming message. Memory unreachable
        # is fatal for this turn (no context, no durable trace) — log and
        # drop; the loop survives.
        try:
            if msg.conversationId is not None:
                existing = await memory.get(msg.conversationId)
                if existing is None:
                    conversation_id = await memory.create()
                    history: list[Message] = []
                else:
                    conversation_id = msg.conversationId
                    history = list(existing.messages)
            else:
                conversation_id = await memory.create()
                history = []

            user_message = Message(role=Role.user, content=msg.content)
            await tool_runner.run(
                new_call(TOOL_MEMORY_APPEND_ENTRY),
                ToolContext(
                    inputs={
                        "conversation_id": conversation_id,
                        "entry": turn.build_memory_entry(
                            conversation_id=conversation_id,
                            person_id=effective_person_id,
                            message=user_message,
                            nt_snapshot=nt_at_start,
                        ),
                    }
                ),
            )
        except httpx.HTTPError as e:
            log.warning(
                "memory unreachable handling message %s: %s — dropping turn",
                event.eventId,
                e,
            )
            return

        # Recent turns with this person (relationship-context enrichment).
        recent_with_person: list = []
        if effective_person_id != NIL_PERSON_ID:
            try:
                invocation = await tool_runner.run(
                    new_call(TOOL_MEMORY_PERSON_RECENT),
                    ToolContext(
                        inputs={
                            "person_id": effective_person_id,
                            "limit": int(store.get("personRecentLimit") or 30),
                        }
                    ),
                )
                recent_with_person = [
                    e for e in invocation.payload if e.conversationId != conversation_id
                ]
            except httpx.HTTPError as e:
                log.warning("person_recent unreachable: %s (continuing without it)", e)

        cross_pass_framing = str(store.get("crossPassFraming") or "parallel_thread")
        try:
            system_prompts = await turn.build_per_driver_system_prompts(
                drivers=drivers,
                tool_runner=tool_runner,
                person_id=(None if msg.personId == NIL_PERSON_ID else msg.personId),
                operator_person_id=operator_person_id,
                fallback_default=str(store.get("defaultSystemPrompt") or ""),
                recent_turns=recent_with_person,
                cross_pass_framing=cross_pass_framing,
            )
        except httpx.HTTPError as e:
            log.warning("identity unreachable assembling prompts: %s (using default)", e)
            fallback = str(store.get("defaultSystemPrompt") or "")
            system_prompts = {drivers[0].name: fallback, drivers[1].name: fallback}

        history_for_drivers = turn.build_history(history, msg.content)

        # NT no longer modulates the pass count (the plateau-stop gate
        # replaces `modulated_max_passes` in a later increment); for now
        # the deliberative thought runs to the configured cap. Temperature
        # IS still NT-modulated (a parameter, not a limit).
        max_passes = int(store.get("defaultMaxPasses") or 3)
        agreement_threshold = float(store.get("agreementThreshold") or 0.5)
        temp_cfg = store.get("defaultTemperature")
        temperature = modulated_temperature(
            nt_at_start, float(temp_cfg) if temp_cfg is not None else None
        )
        max_tokens_cfg = store.get("defaultMaxTokens")
        max_tokens = int(max_tokens_cfg) if max_tokens_cfg is not None else None

        # THINK — the deliberative flavor of a thought is the bicameral pair.
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
                scorer=scorer,
                cross_pass_framing=cross_pass_framing,
            )
        except (HemisphereDriverError, httpx.HTTPError) as e:
            log.warning("deliberation failed for message %s: %s — no reply", event.eventId, e)
            return

        for pass_record in outcome.passes:
            self._publish("thought", pass_record.model_dump(mode="json", exclude_none=True))

        # GATE — deterministic for 2a: a message that produced a
        # deliberation is spoken. The NT-valence accumulator replaces this.
        self._publish_gate("speak", focus=new_focus)

        # SPEAK — the voice pass IS the speech effector.
        voice_driver = turn.resolve_voice_driver(drivers, store.get("voiceDriver"))
        voice_temp_cfg = store.get("voiceTemperature")
        voice_temperature = float(voice_temp_cfg) if voice_temp_cfg is not None else temperature
        deliberation_finals = outcome.passes[-1].hemispheres if outcome.passes else []
        final_agreement = outcome.passes[-1].callosum.agreement if outcome.passes else 0.5
        voice_persona = system_prompts.get(voice_driver.name) or next(
            iter(system_prompts.values()), ""
        )
        try:
            voice_outcome = await run_voice_pass(
                voice_driver=voice_driver,
                user_message=Message(role=Role.user, content=msg.content),
                history=history,
                system_prompt=voice_persona,
                deliberation_finals=list(deliberation_finals),
                final_agreement=final_agreement,
                nt_state=nt_at_start,
                temperature=voice_temperature,
                max_tokens=max_tokens,
            )
        except (HemisphereDriverError, httpx.HTTPError) as e:
            log.warning("voice pass failed for message %s: %s — no reply", event.eventId, e)
            return

        # Evolve NT from the turn's observations (the `internal` tool). A
        # failure here must NOT discard the already-composed reply — fall
        # back to the pre-turn state and still speak (the same principle the
        # reply-persist step below documents). NTState carries a required
        # `lastUpdated` datetime, so the publish uses mode="json" like every
        # other event — without it `json.dumps` in the SSE route would raise
        # and kill the subscriber's stream.
        observations = turn.build_observations(outcome, agreement_threshold)
        nt_at_end: NTState = nt_at_start
        try:
            nt_invocation = await tool_runner.run(
                new_call(TOOL_NT_OBSERVE),
                ToolContext(inputs={"state": nt_at_start, "observations": observations}),
            )
            nt_at_end = nt_invocation.payload
            app.state.nt_state = nt_at_end
            self._publish("nt_update", nt_at_end.model_dump(mode="json"))
        except Exception:
            log.exception(
                "NT observe failed for message %s — keeping prior state, still replying",
                event.eventId,
            )

        final_message = voice_outcome.output

        # Persist the reply. A persistence failure here is non-fatal — the
        # reply was already decided; log and still speak.
        try:
            await tool_runner.run(
                new_call(TOOL_MEMORY_APPEND_ENTRY),
                ToolContext(
                    inputs={
                        "conversation_id": conversation_id,
                        "entry": turn.build_memory_entry(
                            conversation_id=conversation_id,
                            person_id=effective_person_id,
                            message=final_message,
                            nt_snapshot=nt_at_end,
                            hemisphere_attribution="voice",
                        ),
                    }
                ),
            )
        except httpx.HTTPError as e:
            log.warning("memory unreachable persisting reply to %s: %s", event.eventId, e)

        # Surface the turn's tool trace on the stream (the M0.5 debug lens,
        # now live rather than bundled into a response).
        for record in trace:
            self._publish("tool_call", record.model_dump(mode="json", exclude_none=True))

        # SPEAK effector — emit the utterance. Destination mirrors the
        # afferent source; `inResponseTo` correlates it to the event. For
        # now the speech leaves on the consciousness stream; connector
        # delivery to external channels is a later slice.
        speech = EfferentSpeechAct(
            destination=msg.source,
            content=final_message.content,
            inResponseTo=event.eventId,
            conversationId=conversation_id,
            timestamp=datetime.now(UTC),
        )
        self._publish("speech", speech.model_dump(mode="json", exclude_none=True))

    # -- observability -------------------------------------------------

    def _publish(self, event_type: str, data: dict) -> None:
        self._broker.publish(event_type, data)

    def _publish_gate(self, action: str, *, focus: str | None) -> None:
        decision = GateDecision.model_validate({"action": action, "focus": focus})
        self._broker.publish("gate_decision", decision.model_dump(mode="json", exclude_none=True))
