"""GET /v1/conversations/{id}."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status

from .._generated.models import Conversation, Problem
from ..memory import InProcessMemory

router = APIRouter(tags=["chat"])


@router.get("/v1/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(request: Request, conversation_id: UUID) -> Conversation:
    memory: InProcessMemory = request.app.state.memory
    conversation = memory.get(conversation_id)
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=Problem(
                type="https://github.com/eugene-plexus/orchestrator#conversation-not-found",
                title="Conversation not found",
                status=404,
                detail=f"No conversation with id {conversation_id}.",
                component="orchestrator",
            ).model_dump(exclude_none=True),
        )
    return conversation
