"""The bicameral loop.

For each turn:

1. Build the prompt from conversation history + the new user message.
2. Send to all configured drivers in parallel.
3. Score corpus-callosum agreement on their outputs.
4. If agreement >= threshold, terminate and emit a blended response.
5. Otherwise, append every driver's output + a re-think prompt and run
   another pass, up to `max_passes`.

v0.1 keeps this *deliberately mechanical*: there's no smart prompting
between passes, no per-driver prompt variation, no NT modulation. The
goal is to validate the cross-vendor architecture; meaningful pass-policy
work comes after we have data.

v0.1 also pins the loop at *exactly two* drivers — the agreement scoring
and blend functions are pairwise. v0.2+ generalizes to N (with a real
N-way reconciliation strategy and failover semantics).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from .._generated.hemisphere_models import GenerateRequest
from .._generated.models import (
    CallosumState,
    Decision,
    Message,
    NTState,
    PassRecord,
    Role,
)
from ..hemisphere_client import HemisphereClient
from .callosum import blend, jaccard_word_agreement

log = logging.getLogger(__name__)

REPROMPT_INSTRUCTION = (
    "Your two hemispheres returned divergent responses on the previous pass. "
    "Reconsider, taking both prior responses into account, and produce a "
    "unified, well-reasoned answer."
)


class BicameralPairRequired(RuntimeError):
    """Raised when the orchestrator's `drivers` list is not exactly two.

    v0.1's agreement / blend functions are pairwise; running with a
    different driver count would silently misbehave. v0.2+ will replace
    this guard with a real N-way reconciliation strategy.
    """


@dataclass
class BicameralOutcome:
    """Result of running the bicameral loop for one turn.

    `pass_latencies_ms` is the max-of-hemispheres latency for each
    pass (parallel dispatch → the pass takes as long as the slower
    hemisphere). The NT system reads this to nudge norepinephrine.
    """

    final_message: Message
    passes: list[PassRecord]
    pass_latencies_ms: list[int]


# Max characters per message preview in DEBUG-level traces. Long enough
# for short prompts to render in full, short enough that a multi-pass
# loop with 4K-token hemispheres doesn't drown the log file.
_DEBUG_PREVIEW_CHARS = 400


def _preview(text: str, *, limit: int = _DEBUG_PREVIEW_CHARS) -> str:
    """One-line preview of message content for DEBUG logs. Collapses
    newlines to ⏎ and truncates with a length suffix so the operator can
    see how much they're missing."""
    flat = text.replace("\n", "⏎ ")
    if len(flat) <= limit:
        return flat
    return f"{flat[:limit]}… [+{len(flat) - limit} chars]"


async def run_bicameral_loop(
    *,
    history: list[Message],
    system_prompts: dict[str, str],
    drivers: list[HemisphereClient],
    nt_state: NTState,
    max_passes: int,
    agreement_threshold: float,
    temperature: float | None,
    max_tokens: int | None,
) -> BicameralOutcome:
    """Drive the bicameral loop for a single user turn.

    `history` is the shared conversation history — user / assistant
    messages, no system message. Each driver sees the same history.

    `system_prompts` maps driver name to that driver's system message
    content. v0.2's identity-assembled prompts give each hemisphere a
    distinct preamble identifying which side it is and what model its
    twin is running — see `chat.py::_build_per_driver_system_prompts`.

    `temperature` and `max_tokens` are applied to every `GenerateRequest`
    built here. The orchestrator owns LLM-output-affecting parameters; the
    driver does not substitute defaults of its own. In v0.2+ these will be
    derived per-pass from `nt_state` instead of being supplied as flat
    arguments — until then the caller passes the configured baseline.

    `drivers` carries the operator-supplied driver names; each emitted
    `Message` is stamped with `driverName` so the UI and downstream
    consumers can label outputs without knowing the topology in advance.
    """
    if len(drivers) != 2:
        raise BicameralPairRequired(
            f"v0.1 bicameral loop requires exactly two drivers; got {len(drivers)}. "
            "N-driver reconciliation lands in v0.2+."
        )
    left, right = drivers[0], drivers[1]

    for driver in drivers:
        if driver.name not in system_prompts:
            raise ValueError(
                f"system_prompts is missing an entry for driver {driver.name!r}; "
                f"got keys {sorted(system_prompts)}"
            )

    passes: list[PassRecord] = []
    pass_latencies_ms: list[int] = []
    # `intermediate` carries between-pass content shared across drivers:
    # the hemisphere outputs from prior passes plus the reprompt
    # nudges. Each driver's outgoing message list = [system(theirs)] +
    # history + intermediate.
    intermediate: list[Message] = []

    def _build_messages_for(driver: HemisphereClient) -> list[Message]:
        out: list[Message] = []
        sys_prompt = system_prompts[driver.name]
        if sys_prompt:
            out.append(Message(role=Role.system, content=sys_prompt))
        out.extend(history)
        out.extend(intermediate)
        return out

    def _build_request_for(driver: HemisphereClient, pass_index: int) -> GenerateRequest:
        messages = _build_messages_for(driver)
        request_payload: dict[str, object] = {
            "messages": [m.model_dump(mode="json", exclude_none=True) for m in messages],
            "ntState": nt_state.model_dump(exclude_none=True),
            "passIndex": pass_index,
        }
        if temperature is not None:
            request_payload["temperature"] = temperature
        if max_tokens is not None:
            request_payload["maxTokens"] = max_tokens
        return GenerateRequest.model_validate(request_payload)

    for pass_index in range(max_passes):
        # Build the cross-spec GenerateRequest by serializing through dict.
        # The orchestrator.yaml and hemisphere-driver.yaml each have their
        # own generated Message / NTState classes (same wire shape, distinct
        # Python types because we keep the two model modules independent).
        left_request = _build_request_for(left, pass_index)
        right_request = _build_request_for(right, pass_index)

        log.debug("bicameral pass %d: dispatching to %d drivers", pass_index, len(drivers))
        if log.isEnabledFor(logging.DEBUG):
            for driver, request in ((left, left_request), (right, right_request)):
                for i, m in enumerate(request.messages):
                    log.debug(
                        "  pass %d outgoing[%s][%d] role=%s%s content=%s",
                        pass_index,
                        driver.name,
                        i,
                        m.role.value,
                        f" driver={m.driverName}" if m.driverName else "",
                        _preview(m.content),
                    )

        left_resp, right_resp = await asyncio.gather(
            left.generate(left_request),
            right.generate(right_request),
        )

        # Pass latency = the slower hemisphere's contribution (they ran
        # in parallel; the pass took as long as whichever finished last).
        # Defaults to 0 when a driver doesn't report latency.
        pass_latency = max(left_resp.latencyMs or 0, right_resp.latencyMs or 0)
        pass_latencies_ms.append(pass_latency)

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "  pass %d response left (%s) finish=%s latency=%dms: %s",
                pass_index,
                left.name,
                left_resp.finishReason.value,
                left_resp.latencyMs or 0,
                _preview(left_resp.content),
            )
            log.debug(
                "  pass %d response right (%s) finish=%s latency=%dms: %s",
                pass_index,
                right.name,
                right_resp.finishReason.value,
                right_resp.latencyMs or 0,
                _preview(right_resp.content),
            )

        left_msg = Message(
            role=Role.hemisphere,
            content=left_resp.content,
            driverName=left.name,
            passIndex=pass_index,
        )
        right_msg = Message(
            role=Role.hemisphere,
            content=right_resp.content,
            driverName=right.name,
            passIndex=pass_index,
        )

        score = jaccard_word_agreement(left_resp.content, right_resp.content)
        is_last_pass = pass_index == max_passes - 1
        agreed = score >= agreement_threshold
        log.debug(
            "  pass %d callosum agreement=%.3f threshold=%.2f agreed=%s is_last=%s",
            pass_index,
            score,
            agreement_threshold,
            agreed,
            is_last_pass,
        )

        if agreed or is_last_pass:
            decision = Decision.terminate if agreed else Decision.cap_reached
            blended_text = blend(left_resp.content, right_resp.content)
            blended_msg = Message(
                role=Role.assistant,
                content=blended_text,
                passIndex=pass_index,
            )
            passes.append(
                PassRecord(
                    passIndex=pass_index,
                    hemispheres=[left_msg, right_msg],
                    callosum=CallosumState(
                        agreement=score,
                        decision=decision,
                        blendedMessage=blended_msg,
                    ),
                )
            )
            log.info(
                "bicameral done in %d pass(es), agreement=%.2f, decision=%s",
                pass_index + 1,
                score,
                decision.value,
            )
            log.debug("  pass %d blended: %s", pass_index, _preview(blended_text))
            return BicameralOutcome(
                final_message=blended_msg,
                passes=passes,
                pass_latencies_ms=pass_latencies_ms,
            )

        passes.append(
            PassRecord(
                passIndex=pass_index,
                hemispheres=[left_msg, right_msg],
                callosum=CallosumState(
                    agreement=score,
                    decision=Decision.another_pass,
                ),
            )
        )

        intermediate.append(left_msg)
        intermediate.append(right_msg)
        # REPROMPT is sent as `user` not `system` — many providers
        # (MiniMax-M2.x, some xAI variants, several local OpenAI-compat
        # servers) reject any system message after the leading one with
        # a "invalid message role: system" 400. Semantically the
        # reprompt is the orchestrator-as-user nudging the consciousness
        # to reconsider, so `user` is also the more accurate role.
        intermediate.append(Message(role=Role.user, content=REPROMPT_INSTRUCTION))

    raise RuntimeError(
        "bicameral loop exited without producing a final message — should be unreachable"
    )
