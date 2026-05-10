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
    """Result of running the bicameral loop for one turn."""

    final_message: Message
    passes: list[PassRecord]


async def run_bicameral_loop(
    *,
    initial_messages: list[Message],
    drivers: list[HemisphereClient],
    nt_state: NTState,
    max_passes: int,
    agreement_threshold: float,
    temperature: float | None,
    max_tokens: int | None,
) -> BicameralOutcome:
    """Drive the bicameral loop for a single user turn.

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

    passes: list[PassRecord] = []
    messages: list[Message] = list(initial_messages)

    for pass_index in range(max_passes):
        # Build the cross-spec GenerateRequest by serializing through dict.
        # The orchestrator.yaml and hemisphere-driver.yaml each have their
        # own generated Message / NTState classes (same wire shape, distinct
        # Python types because we keep the two model modules independent).
        request_payload: dict[str, object] = {
            "messages": [m.model_dump(mode="json", exclude_none=True) for m in messages],
            "ntState": nt_state.model_dump(exclude_none=True),
            "passIndex": pass_index,
        }
        if temperature is not None:
            request_payload["temperature"] = temperature
        if max_tokens is not None:
            request_payload["maxTokens"] = max_tokens
        gen_request = GenerateRequest.model_validate(request_payload)

        log.debug("bicameral pass %d: dispatching to %d drivers", pass_index, len(drivers))
        left_resp, right_resp = await asyncio.gather(
            left.generate(gen_request),
            right.generate(gen_request),
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
            return BicameralOutcome(final_message=blended_msg, passes=passes)

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

        messages.append(left_msg)
        messages.append(right_msg)
        messages.append(Message(role=Role.system, content=REPROMPT_INSTRUCTION))

    raise RuntimeError(
        "bicameral loop exited without producing a final message — should be unreachable"
    )
