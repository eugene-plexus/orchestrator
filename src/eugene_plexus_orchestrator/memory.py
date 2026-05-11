"""Conversation memory: protocol + in-process and HTTP implementations.

Production path is `HttpMemory`, which talks to the
[eugene-plexus/memory](https://github.com/eugene-plexus/memory) service
over its OpenAPI surface. `InProcessMemory` is retained as a test double
and dev fallback — tests inject it via `app.state.memory` before the
FastAPI lifespan runs, mirroring how the hemisphere clients are
injected.

Both implementations satisfy the `MemoryClient` Protocol so routes can
type against the contract without caring which backend is in play.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

import httpx

from ._generated.models import Conversation, Message


class MemoryClient(Protocol):
    """Contract every memory backend implements (real or fake-for-tests).

    `append` returns `None` when the conversation does not exist; routes
    map that to a 404 response. Other transport-level failures (the
    memory service is down, a 5xx, etc.) raise `httpx.HTTPError` and the
    route's existing 502 handler picks it up.
    """

    async def create(self) -> UUID: ...
    async def get(self, conversation_id: UUID) -> Conversation | None: ...
    async def append(self, conversation_id: UUID, message: Message) -> Message | None: ...
    async def delete(self, conversation_id: UUID) -> bool: ...
    async def aclose(self) -> None: ...


class InProcessMemory:
    """Test double / dev fallback. Single-process, thread-unaware — fine for
    FastAPI's single-event-loop runtime, not appropriate for production.
    The eugene-plexus/memory service is the production path."""

    def __init__(self) -> None:
        self._conversations: dict[UUID, list[Message]] = {}

    async def create(self) -> UUID:
        cid = uuid4()
        self._conversations[cid] = []
        return cid

    async def get(self, conversation_id: UUID) -> Conversation | None:
        messages = self._conversations.get(conversation_id)
        if messages is None:
            return None
        return Conversation(id=conversation_id, messages=list(messages))

    async def append(self, conversation_id: UUID, message: Message) -> Message | None:
        if conversation_id not in self._conversations:
            return None
        stamped = _ensure_timestamp(message)
        self._conversations[conversation_id].append(stamped)
        return stamped

    async def delete(self, conversation_id: UUID) -> bool:
        return self._conversations.pop(conversation_id, None) is not None

    async def aclose(self) -> None:
        return None


class HttpMemory:
    """HTTP-backed memory client. Talks to eugene-plexus/memory."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 30.0,
        service_token: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        # See HttpHemisphereClient — same pattern: thread the
        # orchestrator's service-audience bearer onto every outbound
        # request to the memory service.
        headers = {"Authorization": f"Bearer {service_token}"} if service_token else None
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            headers=headers,
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    async def create(self) -> UUID:
        response = await self._client.post("/v1/conversations")
        response.raise_for_status()
        conversation = Conversation.model_validate(response.json())
        # The spec marks Conversation.id optional (server-assigned) so the
        # codegen'd type is UUID | None. The memory service always sets it
        # on the create-201 response — assert that contract here so a
        # spec-violating backend surfaces as a loud error rather than a
        # silent None bouncing further down the stack.
        if conversation.id is None:
            raise ValueError(
                "memory service returned a conversation without an id "
                "(violates POST /v1/conversations contract)"
            )
        return conversation.id

    async def get(self, conversation_id: UUID) -> Conversation | None:
        response = await self._client.get(f"/v1/conversations/{conversation_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return Conversation.model_validate(response.json())

    async def append(self, conversation_id: UUID, message: Message) -> Message | None:
        payload = message.model_dump(mode="json", exclude_none=True)
        response = await self._client.post(
            f"/v1/conversations/{conversation_id}/messages",
            json=payload,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return Message.model_validate(response.json())

    async def delete(self, conversation_id: UUID) -> bool:
        response = await self._client.delete(f"/v1/conversations/{conversation_id}")
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True

    async def aclose(self) -> None:
        await self._client.aclose()


def _ensure_timestamp(message: Message) -> Message:
    if message.timestamp is not None:
        return message
    return message.model_copy(update={"timestamp": datetime.now(UTC)})
