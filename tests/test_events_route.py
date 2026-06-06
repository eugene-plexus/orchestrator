"""POST /v1/events — the afferent-event injection door.

Fire-and-forget: accepts an `AfferentEvent`, enqueues it for the
continuous loop, returns 202 with the eventId echoed. The loop's actual
handling is covered by the direct-loop tests (test_tool_trace,
test_nt_chat_evolution, test_memory_personid_wiring); here we only verify
the route wiring and the acceptance contract.
"""

from __future__ import annotations

from uuid import UUID

from fastapi.testclient import TestClient

from tests.conftest import make_message_event


def test_inject_event_returns_202_and_echoes_event_id(client: TestClient) -> None:
    event = make_message_event("hello")
    response = client.post(
        "/v1/events",
        json=event.model_dump(mode="json", exclude_none=True),
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["accepted"] is True
    assert UUID(body["eventId"]) == event.eventId


def test_inject_event_rejects_malformed_payload(client: TestClient) -> None:
    # Missing required fields (kind/source/eventId) → 422 from validation.
    response = client.post("/v1/events", json={"message": {"content": "hi"}})
    assert response.status_code == 422
