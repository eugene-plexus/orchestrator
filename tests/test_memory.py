"""Tests for the in-process memory store."""

from __future__ import annotations

from uuid import uuid4

from eugene_plexus_orchestrator._generated.models import Message, Role
from eugene_plexus_orchestrator.memory import InProcessMemory


def test_create_then_get_returns_empty_conversation() -> None:
    mem = InProcessMemory()
    cid = mem.create()
    convo = mem.get(cid)
    assert convo is not None
    assert convo.id == cid
    assert convo.messages == []


def test_get_unknown_returns_none() -> None:
    mem = InProcessMemory()
    assert mem.get(uuid4()) is None


def test_append_persists_and_stamps_timestamp() -> None:
    mem = InProcessMemory()
    cid = mem.create()
    written = mem.append(cid, Message(role=Role.user, content="hi"))
    assert written.timestamp is not None
    convo = mem.get(cid)
    assert convo is not None and convo.messages[0].content == "hi"


def test_delete_removes() -> None:
    mem = InProcessMemory()
    cid = mem.create()
    assert mem.delete(cid) is True
    assert mem.get(cid) is None
    assert mem.delete(cid) is False
