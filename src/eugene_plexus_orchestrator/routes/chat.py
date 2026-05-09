"""POST /v1/chat (real) and POST /v1/chat/stream (still 501)."""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException, Request, status

from .._generated.models import (
    ChatRequest,
    ChatResponse,
    Message,
    Problem,
    Role,
)
from ..bicameral.loop import run_bicameral_loop
from ..bicameral.nt import neutral_state
from ..config import ConfigStore
from ..hemisphere_client import HemisphereClient
from ..memory import InProcessMemory

router = APIRouter(tags=["chat"])

log = logging.getLogger(__name__)


def _build_initial_messages(
    history: list[Message], system_prompt: str, user_message: str
) -> list[Message]:
    """Build the messages list to send to hemispheres for this turn."""
    out: list[Message] = []
    if system_prompt:
        out.append(Message(role=Role.system, content=system_prompt))
    # Include only user/assistant turns from history; hemisphere-tagged
    # intermediate messages from prior turns are observability artifacts
    # and don't belong in subsequent prompts.
    for msg in history:
        if msg.role in (Role.user, Role.assistant):
            out.append(msg)
    out.append(Message(role=Role.user, content=user_message))
    return out


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    store: ConfigStore = request.app.state.config_store
    memory: InProcessMemory = request.app.state.memory
    left: HemisphereClient = request.app.state.left_driver
    right: HemisphereClient = request.app.state.right_driver

    if body.conversationId is not None:
        existing = memory.get(body.conversationId)
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
        conversation_id = memory.create()
        history = []

    system_prompt = body.systemPrompt or str(store.get("defaultSystemPrompt") or "")
    user_message = Message(role=Role.user, content=body.message)
    memory.append(conversation_id, user_message)

    initial_messages = _build_initial_messages(history, system_prompt, body.message)

    nt_at_start = neutral_state()
    max_passes = int(body.maxPasses or store.get("defaultMaxPasses") or 3)
    agreement_threshold = float(store.get("agreementThreshold") or 0.5)

    try:
        outcome = await run_bicameral_loop(
            initial_messages=initial_messages,
            left=left,
            right=right,
            nt_state=nt_at_start,
            max_passes=max_passes,
            agreement_threshold=agreement_threshold,
        )
    except httpx.HTTPError as e:
        log.warning("bicameral loop failed at HTTP layer: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=Problem(
                type="https://github.com/eugene-plexus/orchestrator#hemisphere-error",
                title="Hemisphere driver error",
                status=502,
                detail=str(e),
                component="orchestrator",
            ).model_dump(exclude_none=True),
        ) from e

    memory.append(conversation_id, outcome.final_message)

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
