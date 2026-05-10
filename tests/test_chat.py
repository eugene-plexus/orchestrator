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
    # Hemisphere messages carry the operator-supplied driver name.
    driver_names = [m["driverName"] for m in body["passes"][0]["hemispheres"]]
    assert driver_names == ["left", "right"]


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


def test_chat_502_propagates_driver_problem_detail(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """When a driver returns a structured Problem (e.g. OpenAI rejected
    a parameter), the orchestrator surfaces its title + detail in the
    response — not the generic httpx '502 Bad Gateway' string. Without
    this, debugging upstream model errors means tailing driver logs
    every time."""
    from eugene_plexus_orchestrator._generated.hemisphere_models import Problem
    from eugene_plexus_orchestrator.hemisphere_client import HemisphereDriverError

    upstream_detail = (
        "openai_api returned 400: Unsupported value: 'temperature' does not "
        "support 0.7 with this model. Only the default (1) value is supported."
    )
    right_fake.generate_error = HemisphereDriverError(
        driver_name=right_fake.name,
        driver_url=right_fake.base_url,
        status_code=502,
        problem=Problem(
            type="https://github.com/eugene-plexus/hemisphere-driver#cli-error",
            title="Backend CLI error",
            status=502,
            detail=upstream_detail,
            component="hemisphere-driver:openai_api",
        ),
        raw_body="{}",
    )

    response = client.post("/v1/chat", json={"message": "hi"})
    assert response.status_code == 502
    body = response.json()
    detail = body["detail"]
    # Driver name + upstream title appear in the orchestrator's title.
    assert "right" in detail["title"]
    assert "Backend CLI error" in detail["title"]
    # Upstream detail (the actual root cause) appears verbatim.
    assert "temperature" in detail["detail"]
    assert "0.7" in detail["detail"]
    # Provenance is preserved so an operator can trace the error.
    assert "hemisphere-driver:openai_api" in detail["detail"]


def test_chat_502_falls_back_to_raw_body_when_driver_returns_non_problem(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """If the driver's error response isn't parseable as Problem JSON,
    the orchestrator falls back to including the raw body — never just
    swallows it into a generic message."""
    from eugene_plexus_orchestrator.hemisphere_client import HemisphereDriverError

    right_fake.generate_error = HemisphereDriverError(
        driver_name=right_fake.name,
        driver_url=right_fake.base_url,
        status_code=500,
        problem=None,
        raw_body="<html>500 Internal Server Error from a misbehaving proxy</html>",
    )
    response = client.post("/v1/chat", json={"message": "hi"})
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "misbehaving proxy" in detail["detail"]
    assert "upstream-status=500" in detail["detail"]


def test_chat_sets_temperature_and_max_tokens_from_config_on_every_pass(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """LLM-output-affecting params are owned by the orchestrator. Every
    GenerateRequest the driver sees must carry them — sourced from
    orchestrator config in v0.1, from NT state in v0.2+. Drivers may not
    fall back to local defaults."""
    # Override the defaults so we know any value present came from config,
    # not a request-construction fallback.
    patch = client.patch(
        "/v1/config",
        json={"defaultTemperature": 0.3, "defaultMaxTokens": 512},
    )
    assert patch.status_code == 200, patch.text

    left_fake.responses = ["alpha", "beta"]
    right_fake.responses = ["gamma", "beta"]

    response = client.post("/v1/chat", json={"message": "hi", "maxPasses": 2})
    assert response.status_code == 200, response.text

    # Both hemispheres got at least one call. Every recorded GenerateRequest
    # — across passes — must carry the configured defaults.
    assert left_fake.calls and right_fake.calls
    for req in left_fake.calls + right_fake.calls:
        assert req.temperature == 0.3
        assert req.maxTokens == 512
