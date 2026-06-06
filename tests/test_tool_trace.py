"""M0.5: the chat response carries a tool-invocation trace.

Every operation the orchestrator performs runs through the ToolRunner,
and each invocation is recorded on `ChatResponse.toolInvocations` so the
UI can render the perception/action layer beneath deliberation — the
primary debug surface for the tool-calling substrate.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import FakeHemisphereClient


def test_chat_response_carries_tool_invocation_trace(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    # A body personId makes the afferent memory recall fire (it's skipped
    # for the NIL person), so this single turn exercises all three
    # channels even with identity off.
    person_id = "11111111-1111-1111-1111-111111111111"
    response = client.post("/v1/chat", json={"message": "hello", "personId": person_id})
    assert response.status_code == 200, response.text
    body = response.json()

    invocations = body["toolInvocations"]
    assert invocations, "expected a non-empty tool-invocation trace"

    names = {inv["name"] for inv in invocations}
    assert "memory_person_recent" in names  # afferent read
    assert "memory_append_entry" in names  # efferent write
    assert "nt_observe" in names  # internal regimented call

    # Phase-1 turn touches exactly these three channels.
    channels = {inv["channel"] for inv in invocations}
    assert channels == {"afferent", "efferent", "internal"}

    # Reversibility class rides along for the System-1/2 gate; memory
    # writes are reversible, reads/internal are read_only.
    for inv in invocations:
        if inv["name"] == "memory_append_entry":
            assert inv["effect"] == "reversible"
        else:
            assert inv["effect"] == "read_only"

    # Each record is fully shaped for the UI trace.
    for inv in invocations:
        assert inv["name"]
        assert inv["channel"] in {"afferent", "efferent", "internal"}
        assert "latencyMs" in inv


def test_tool_trace_does_not_bleed_across_turns(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """Each turn starts a fresh trace — the second turn must not carry the
    first turn's invocations (guards the task-local contextvar reset
    behavior)."""
    left_fake.responses = ["hi", "hi"]
    right_fake.responses = ["hi", "hi"]

    first = client.post("/v1/chat", json={"message": "one"}).json()
    second = client.post("/v1/chat", json={"message": "two"}).json()

    # Identity off + NIL person → no afferent recall; both turns run the
    # same small set (two efferent writes + one internal nt_observe).
    assert len(second["toolInvocations"]) == len(first["toolInvocations"])
    assert {i["name"] for i in second["toolInvocations"]} == {
        "memory_append_entry",
        "nt_observe",
    }
