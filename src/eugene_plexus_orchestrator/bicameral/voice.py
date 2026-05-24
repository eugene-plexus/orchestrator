"""The voice pass — post-deliberation user-facing reply generation.

The bicameral deliberation loop in `loop.py` produces hemispheres
that talk to each other to reconcile. That works for internal
deliberation but produces a register problem: by pass 1+, each
hemisphere is responding to its OTHER half ("Glad we're on the same
page," "Thanks for not contradicting me"), not to the actual user.
The last hemisphere output therefore isn't appropriate to show as
Eugene's reply — it's deliberation overheard, not communication.

The voice pass is one additional LLM call after the deliberation
loop terminates. It receives:
  - Eugene's persona prompt (the same one used during deliberation)
  - The conversation history
  - The user's current message
  - An inline summary of "what you just considered" — both
    hemispheres' final deliberated content
  - A directive: "now respond to the user. Not to yourself."

The voice pass's output IS what the user sees. The hemispheres'
raw outputs stop being user-facing — they become internal
artifacts the voice pass draws from. This also implicitly solves
the `<think>`-leak risk (the voice pass regenerates clean text
regardless of how messy the deliberation got).

v0.2.x: voice pass always runs. Operator picks which driver does
it via `voiceDriver` config. v0.3+ can add NT modulation of voice
temperature, multi-voice (chorus), or skip-the-voice-pass under
certain NT states.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .._generated.hemisphere_models import GenerateRequest
from .._generated.models import (
    Message,
    NTState,
    Role,
)
from ..hemisphere_client import HemisphereClient

log = logging.getLogger(__name__)

VOICE_PASS_SCRATCHPAD_SUFFIX = (
    "\n\n---\n\n"
    "Internal scratchpad — your two trains of thought during this turn. "
    "These are YOUR private thoughts. The user can't see this and you "
    "shouldn't address it or recite it back:\n\n"
    "{deliberation_summary}\n\n"
    "---\n\n"
    "When responding to the user: react first. If their message is "
    "confusing, weird, gibberish, a non-sequitur, or a strange opener — "
    "react to THAT, briefly. Real people don't owe strangers a coherent "
    "on-topic answer. Don't explain, don't catalog, don't lecture. "
    "You can be confused, dismissive, playful, defensive, curious, "
    "terse, or thrown off the rails. Keep it short unless the user "
    "actually asks for more."
)
# v0.2.x smoke-test structure. The prior layout sent two user-role
# messages (the actual user prompt + a trailing directive containing
# the deliberation summary). The model treated the second as
# something to respond to — which is why post-fix output ALWAYS
# contained an architecture-leak observation ("you just narrated my
# internal monologue back at me") followed by an essay covering
# everything the deliberation touched.
#
# New structure: scratchpad goes into the system message as the
# model's PRIVATE notes, and the user-role input is just the actual
# user message. The model now generates a reply to a single user
# turn, informed by but not addressing the scratchpad.
#
# Like the prior variant, this CANNOT live here unconditionally —
# the directive presumes "stranger, first turn, weird opener" social
# context. v0.3 work: pick a scratchpad/directive variant based on
# (a) person familiarity from identity, (b) conversation turn count,
# maybe (c) NT state. For now we're testing whether the structural
# fix + sharper directive language actually moves the
# helpful-agent ceiling.


@dataclass(frozen=True)
class VoicePassOutcome:
    """Result of running the voice pass for one turn."""

    driver_name: str
    input_messages: list[Message]
    output: Message
    latency_ms: int


def _format_deliberation_summary(deliberation_finals: list[Message]) -> str:
    """Render the deliberation's last-pass outputs as a labeled
    bullet list for the voice pass's prompt.

    `deliberation_finals` is one Message per hemisphere — the LAST
    response each side produced during the loop. Their driverName
    is used as a label so the voice pass sees which thread is which,
    even though the user-facing reply won't mention them.
    """
    if not deliberation_finals:
        return "(no deliberation summary — direct voice pass)"
    lines: list[str] = []
    for msg in deliberation_finals:
        label = msg.driverName or "thread"
        lines.append(f"- ({label}) {msg.content.strip()}")
    return "\n".join(lines)


async def run_voice_pass(
    *,
    voice_driver: HemisphereClient,
    user_message: Message,
    history: list[Message],
    system_prompt: str,
    deliberation_finals: list[Message],
    nt_state: NTState,
    temperature: float | None,
    max_tokens: int | None,
) -> VoicePassOutcome:
    """Convert deliberation into Eugene's user-facing reply.

    The voice pass uses the SAME persona/system prompt as deliberation
    (Eugene-as-a-person, not Eugene-the-assistant). What's different
    is the LAST user-role message: it carries an inline summary of the
    just-completed deliberation and instructs the model to address the
    actual user.

    `history` is the conversation history WITHOUT the current
    `user_message` — the current message is appended after history,
    then the voice directive is bracketed in a final user message.
    """
    summary = _format_deliberation_summary(deliberation_finals)
    # Scratchpad lives in the system message as the model's private
    # notes. Without this restructure the model treats the trailing
    # directive as another user turn to respond to and explainers
    # the deliberation summary back at the user verbatim.
    scratchpad_suffix = VOICE_PASS_SCRATCHPAD_SUFFIX.format(
        deliberation_summary=summary
    )
    voice_system_prompt = (system_prompt or "") + scratchpad_suffix

    input_messages: list[Message] = []
    if voice_system_prompt:
        input_messages.append(Message(role=Role.system, content=voice_system_prompt))
    input_messages.extend(history)
    input_messages.append(user_message)

    request_payload: dict[str, object] = {
        "messages": [m.model_dump(mode="json", exclude_none=True) for m in input_messages],
        "ntState": nt_state.model_dump(exclude_none=True),
        # Voice pass is always pass index 0 from the driver's POV —
        # this isn't a deliberation pass.
        "passIndex": 0,
    }
    if temperature is not None:
        request_payload["temperature"] = float(temperature)
    if max_tokens is not None:
        request_payload["maxTokens"] = max_tokens

    request = GenerateRequest.model_validate(request_payload)

    started = time.monotonic()
    response = await voice_driver.generate(request)
    latency_ms = response.latencyMs or int((time.monotonic() - started) * 1000)

    output = Message(
        role=Role.assistant,
        content=response.content,
        driverName=voice_driver.name,
    )

    log.info(
        "voice pass: driver=%s latency=%dms output_len=%d",
        voice_driver.name,
        latency_ms,
        len(response.content),
    )
    return VoicePassOutcome(
        driver_name=voice_driver.name,
        input_messages=input_messages,
        output=output,
        latency_ms=latency_ms,
    )


__all__ = ["VoicePassOutcome", "run_voice_pass"]
