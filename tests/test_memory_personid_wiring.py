"""Orchestrator-side v0.2 memory integration tests.

These verify the contract changes from the memory-upgrade phase:

  - Chat handler writes a full MemoryEntry (personId, NT snapshot,
    hemisphereAttribution="blended" on the reply) — not a bare Message.
  - When identity resolves an operator personId, both writes for the
    turn use that personId.
  - body.personId from the request body wins over the operator fallback.
  - When neither identity nor body.personId provide a personId, writes
    fall back to NIL_PERSON_ID without failing the chat turn.
  - `person_recent` is consulted only for non-NIL personIds, and the
    active conversation is excluded from the recent-turns context
    (those turns are already in `history`).
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from eugene_plexus_orchestrator._generated.models import (
    Constitution,
    MemoryEntry,
    Person,
    Role,
)
from eugene_plexus_orchestrator.app import create_app
from eugene_plexus_orchestrator.identity import InProcessIdentity
from eugene_plexus_orchestrator.memory import NIL_PERSON_ID, InProcessMemory
from eugene_plexus_orchestrator.settings import Settings
from tests.conftest import FakeHemisphereClient


def test_chat_writes_full_memory_entries_with_resolved_personid(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """When identity is wired, both writes (user turn + Eugene's reply)
    carry the operator's personId, an NT snapshot, and the reply is
    tagged `hemisphereAttribution: blended`."""
    operator_id = uuid4()
    operator = Person(
        personId=operator_id,
        displayName="Troy",
        isOperator=True,
        createdAt=datetime.now(UTC),
    )

    app = create_app(settings=settings)
    app.state.drivers = [left_fake, right_fake]
    memory = InProcessMemory()
    app.state.memory = memory
    app.state.memory_url = "in-process"
    app.state.identity = InProcessIdentity(persons=[operator])
    app.state.identity_url = "in-process"

    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    with TestClient(app) as client:
        response = client.post("/v1/chat", json={"message": "hello"})
    assert response.status_code == 200, response.text

    cid = UUID(response.json()["conversationId"])
    entries = memory._conversations[cid]
    assert len(entries) == 2

    user_entry, reply_entry = entries
    assert user_entry.personId == operator_id
    assert reply_entry.personId == operator_id
    assert user_entry.role == Role.user
    assert reply_entry.role == Role.assistant
    assert reply_entry.hemisphereAttribution == "blended"
    # NT snapshot present on both — gives the v0.3+ analyser something
    # to correlate output style against state.
    assert user_entry.ntStateSnapshot is not None
    assert reply_entry.ntStateSnapshot is not None


def test_chat_body_personid_wins_over_operator_fallback(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """Connector adapters supply body.personId. Even when an operator
    is registered, that explicit personId must win — the speaker isn't
    the operator."""
    operator_id = uuid4()
    other_id = uuid4()
    operator = Person(
        personId=operator_id,
        displayName="Troy",
        isOperator=True,
        createdAt=datetime.now(UTC),
    )
    other = Person(
        personId=other_id,
        displayName="Sarah",
        isOperator=False,
        createdAt=datetime.now(UTC),
    )

    app = create_app(settings=settings)
    app.state.drivers = [left_fake, right_fake]
    memory = InProcessMemory()
    app.state.memory = memory
    app.state.memory_url = "in-process"
    app.state.identity = InProcessIdentity(persons=[operator, other])
    app.state.identity_url = "in-process"

    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={"message": "hello", "personId": str(other_id)},
        )
    assert response.status_code == 200, response.text

    cid = UUID(response.json()["conversationId"])
    entries = memory._conversations[cid]
    assert all(e.personId == other_id for e in entries)


def test_chat_falls_back_to_NIL_personid_when_no_identity_and_no_body(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """No identity component + no body.personId = NIL_PERSON_ID. Chat
    still succeeds — orchestrator never fails the turn over personId
    resolution alone."""
    app = create_app(settings=settings)
    app.state.drivers = [left_fake, right_fake]
    memory = InProcessMemory()
    app.state.memory = memory
    app.state.memory_url = "in-process"
    app.state.identity = None
    app.state.identity_url = ""

    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    with TestClient(app) as client:
        response = client.post("/v1/chat", json={"message": "hello"})
    assert response.status_code == 200, response.text

    cid = UUID(response.json()["conversationId"])
    entries = memory._conversations[cid]
    assert all(e.personId == NIL_PERSON_ID for e in entries)


def test_chat_injects_recent_turns_from_prior_conversations(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """Memory's recent turns with the speaker get injected into the
    hemisphere prompts — concrete context, not just identity's
    summary. Turns from the CURRENT conversation are excluded (already
    in `history`)."""
    operator_id = uuid4()
    operator = Person(
        personId=operator_id,
        displayName="Troy",
        isOperator=True,
        createdAt=datetime.now(UTC),
    )

    memory = InProcessMemory()
    # Pre-seed a prior conversation with three turns from the operator.
    prior_cid = uuid4()
    memory._conversations[prior_cid] = [
        MemoryEntry(
            entryId=uuid4(),
            personId=operator_id,
            conversationId=prior_cid,
            role=Role.user,
            content="we talked about cats last week",
            timestamp=datetime.now(UTC),
        ),
        MemoryEntry(
            entryId=uuid4(),
            personId=operator_id,
            conversationId=prior_cid,
            role=Role.assistant,
            content="and I mentioned my favorite species",
            timestamp=datetime.now(UTC),
        ),
    ]

    app = create_app(settings=settings)
    app.state.drivers = [left_fake, right_fake]
    app.state.memory = memory
    app.state.memory_url = "in-process"
    app.state.identity = InProcessIdentity(
        constitution=Constitution(name="Eugene"),
        persons=[operator],
    )
    app.state.identity_url = "in-process"

    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    with TestClient(app) as client:
        response = client.post("/v1/chat", json={"message": "remember?"})
    assert response.status_code == 200, response.text

    # System message in each hemisphere's GenerateRequest carries the
    # prior-conversation recap.
    for fake in (left_fake, right_fake):
        sys_msg = next(
            m.content for m in fake.calls[0].messages if m.role == Role.system
        )
        assert "cats last week" in sys_msg
        assert "favorite species" in sys_msg
        assert "Recent turns with this person" in sys_msg


def test_chat_skips_recent_turns_for_nil_person(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """When personId resolves to NIL, the orchestrator must NOT call
    `person_recent` — that would surface unrelated NIL-bucket entries
    from prior anonymous turns, which would be confusing rather than
    helpful."""
    memory = InProcessMemory()
    # Pre-seed a NIL-bucket entry from a prior anonymous turn.
    prior_cid = uuid4()
    memory._conversations[prior_cid] = [
        MemoryEntry(
            entryId=uuid4(),
            personId=NIL_PERSON_ID,
            conversationId=prior_cid,
            role=Role.user,
            content="UNRELATED_NIL_BUCKET_LEAK_CANARY",
            timestamp=datetime.now(UTC),
        ),
    ]

    app = create_app(settings=settings)
    app.state.drivers = [left_fake, right_fake]
    app.state.memory = memory
    app.state.memory_url = "in-process"
    app.state.identity = None
    app.state.identity_url = ""

    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    with TestClient(app) as client:
        response = client.post("/v1/chat", json={"message": "hello"})
    assert response.status_code == 200, response.text

    for fake in (left_fake, right_fake):
        sys_msg = next(
            m.content for m in fake.calls[0].messages if m.role == Role.system
        )
        assert "UNRELATED_NIL_BUCKET_LEAK_CANARY" not in sys_msg
