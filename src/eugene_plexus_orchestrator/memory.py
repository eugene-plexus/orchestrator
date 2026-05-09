"""In-process conversation memory.

v0.1 implements the same shape as the (future) `memory` component will, so
swapping it for an HTTP-backed real component is a mechanical change. We
deliberately do *not* expose the orchestrator-internal store as the source
of truth for cross-conversation state; that's the future memory
component's job.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from uuid import UUID, uuid4

from ._generated.models import Conversation, Message


class InProcessMemory:
    """Thread-safe in-memory store of conversations keyed by uuid."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conversations: dict[UUID, list[Message]] = {}

    def create(self) -> UUID:
        cid = uuid4()
        with self._lock:
            self._conversations[cid] = []
        return cid

    def exists(self, conversation_id: UUID) -> bool:
        with self._lock:
            return conversation_id in self._conversations

    def get(self, conversation_id: UUID) -> Conversation | None:
        with self._lock:
            messages = self._conversations.get(conversation_id)
            if messages is None:
                return None
            return Conversation(id=conversation_id, messages=list(messages))

    def append(self, conversation_id: UUID, message: Message) -> Message:
        stamped = _ensure_timestamp(message)
        with self._lock:
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = []
            self._conversations[conversation_id].append(stamped)
        return stamped

    def append_many(self, conversation_id: UUID, messages: Iterable[Message]) -> None:
        for msg in messages:
            self.append(conversation_id, msg)

    def delete(self, conversation_id: UUID) -> bool:
        with self._lock:
            return self._conversations.pop(conversation_id, None) is not None


def _ensure_timestamp(message: Message) -> Message:
    if message.timestamp is not None:
        return message
    return message.model_copy(update={"timestamp": datetime.now(UTC)})
