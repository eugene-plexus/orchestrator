"""Integration tests for POST /v1/chat using fake hemisphere clients."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import FakeHemisphereClient


def test_chat_round_trip_when_hemispheres_agree(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    left_fake.responses = ["hello world"]
    right_fake.responses = ["hello world"]

    response = client.post("/v1/chat", json={"message": "hi"})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["message"]["role"] == "assistant"
    assert body["message"]["content"] == "hello world"
    assert body["conversationId"]
    assert len(body["passes"]) == 1
    assert body["passes"][0]["callosum"]["decision"] == "terminate"
    assert body["passes"][0]["callosum"]["agreement"] == 1.0


def test_chat_runs_more_passes_until_agreement(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    # Pass 0: total disagreement (no shared words). Pass 1: identical.
    left_fake.responses = ["alpha beta", "consensus reached"]
    right_fake.responses = ["gamma delta", "consensus reached"]

    response = client.post("/v1/chat", json={"message": "hi"})
    assert response.status_code == 200, response.text
    body = response.json()

    assert len(body["passes"]) == 2
    assert body["passes"][0]["callosum"]["decision"] == "another_pass"
    assert body["passes"][1]["callosum"]["decision"] == "terminate"
    assert body["message"]["content"] == "consensus reached"


def test_chat_caps_at_max_passes(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    # Always disagree; cap should kick in.
    left_fake.responses = ["a a a", "b b b", "c c c"]
    right_fake.responses = ["x x x", "y y y", "z z z"]

    response = client.post("/v1/chat", json={"message": "hi", "maxPasses": 3})
    assert response.status_code == 200, response.text
    body = response.json()

    assert len(body["passes"]) == 3
    assert body["passes"][-1]["callosum"]["decision"] == "cap_reached"
    # The blended message is the final pass's blend (longer of the two).
    assert body["message"]["content"]


def test_chat_creates_new_conversation_when_id_omitted(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    response = client.post("/v1/chat", json={"message": "hi"})
    convo_id = response.json()["conversationId"]
    assert convo_id

    # Conversation now contains the user turn + the assistant turn.
    fetched = client.get(f"/v1/conversations/{convo_id}").json()
    roles = [m["role"] for m in fetched["messages"]]
    assert roles == ["user", "assistant"]


def test_chat_404s_when_unknown_conversation_id_supplied(client: TestClient) -> None:
    response = client.post(
        "/v1/chat",
        json={
            "message": "hi",
            "conversationId": "00000000-0000-0000-0000-000000000000",
        },
    )
    assert response.status_code == 404


def test_chat_stream_still_returns_501(client: TestClient) -> None:
    response = client.post("/v1/chat/stream", json={"message": "hi"})
    assert response.status_code == 501
