"""Integration tests for POST /v1/chat using fake hemisphere clients."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import FakeHemisphereClient


def test_chat_round_trip_when_hemispheres_agree(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    # v0.2.x adds a voice pass after deliberation — one extra call to
    # the voice driver (left by default). The voice pass output IS what
    # the user sees; deliberation outputs are diagnostic only.
    left_fake.responses = ["hello world", "voice reply"]
    right_fake.responses = ["hello world"]

    response = client.post("/v1/chat", json={"message": "hi"})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["message"]["role"] == "assistant"
    # User-facing reply comes from the voice pass.
    assert body["message"]["content"] == "voice reply"
    assert body["conversationId"]
    assert len(body["passes"]) == 1
    assert body["passes"][0]["callosum"]["decision"] == "terminate"
    assert body["passes"][0]["callosum"]["agreement"] == 1.0
    # Hemisphere deliberation messages carry the operator-supplied
    # driver name and the original deliberation content.
    driver_names = [m["driverName"] for m in body["passes"][0]["hemispheres"]]
    assert driver_names == ["left", "right"]
    for m in body["passes"][0]["hemispheres"]:
        assert m["content"] == "hello world"
    # Voice pass record is surfaced for diagnostic transparency.
    assert body["voicePass"]["driverName"] == "left"
    assert body["voicePass"]["output"]["content"] == "voice reply"


def test_chat_runs_more_passes_until_agreement(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    # Pass 0: total disagreement (no shared words). Pass 1: identical.
    # Pass 2: voice pass (left only).
    left_fake.responses = ["alpha beta", "consensus reached", "voice-final"]
    right_fake.responses = ["gamma delta", "consensus reached"]

    response = client.post("/v1/chat", json={"message": "hi"})
    assert response.status_code == 200, response.text
    body = response.json()

    assert len(body["passes"]) == 2
    assert body["passes"][0]["callosum"]["decision"] == "another_pass"
    assert body["passes"][1]["callosum"]["decision"] == "terminate"
    # The user-facing message is the voice pass output, not the
    # deliberation's last-pass hemisphere reply.
    assert body["message"]["content"] == "voice-final"


def test_chat_caps_at_max_passes(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    # Always disagree; cap should kick in. Extra entry on left for
    # the voice pass that follows the deliberation loop.
    left_fake.responses = ["a a a", "b b b", "c c c", "voice fallback"]
    right_fake.responses = ["x x x", "y y y", "z z z"]

    response = client.post("/v1/chat", json={"message": "hi", "maxPasses": 3})
    assert response.status_code == 200, response.text
    body = response.json()

    assert len(body["passes"]) == 3
    assert body["passes"][-1]["callosum"]["decision"] == "cap_reached"
    # The user-facing message is the voice pass output.
    assert body["message"]["content"] == "voice fallback"


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


def test_agreement_directive_bands_distinct() -> None:
    """The agreement-to-register helper must return materially different
    directives across its three bands.

    Regression guard for the architectural-payoff change: the voice
    pass's whole point of seeing the agreement score is that high vs.
    low agreement produces a different register in the user-facing
    reply. If all three bands collapsed to the same text, the
    bicameral signal would be silent at the user-facing surface
    again."""
    from eugene_plexus_orchestrator.bicameral.voice import _agreement_directive

    high = _agreement_directive(0.9)
    mid = _agreement_directive(0.55)
    low = _agreement_directive(0.2)

    # All three are non-empty.
    assert high.strip() and mid.strip() and low.strip()
    # All three are distinct.
    assert high != mid
    assert mid != low
    assert high != low
    # Band-typical vocabulary appears in the right places.
    assert "conviction" in high.lower() or "certainty" in high.lower()
    assert "friction" in mid.lower() or "imperfect" in mid.lower()
    assert "two minds" in low.lower() or "ambivalent" in low.lower()


def test_chat_voice_pass_scratchpad_carries_low_agreement_directive(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """End-to-end: when deliberation ends with low agreement, the voice
    pass system prompt must include the low-band directive language.

    This is what makes the bicameral architecture's `agreement` score
    visible at the user-facing surface. Without this wire-up, the
    score is internal trivia."""
    # Force disagreement: hemispheres produce non-overlapping content
    # on every pass. Voice pass receives the divergent finals + a
    # low-band register directive.
    left_fake.responses = ["alpha alpha alpha", "beta beta beta", "voice"]
    right_fake.responses = ["xenon xenon xenon", "yttrium yttrium yttrium"]

    response = client.post("/v1/chat", json={"message": "hi", "maxPasses": 2})
    assert response.status_code == 200, response.text

    # The voice pass is the LEFT fake's last call. Its system prompt
    # should contain the low-agreement directive phrasing.
    voice_call = left_fake.calls[-1]
    system_msgs = [m for m in voice_call.messages if m.role.value == "system"]
    assert system_msgs, "voice pass had no system message"
    system_text = system_msgs[0].content
    # Low-band signal words.
    assert "two minds" in system_text.lower() or "did NOT agree" in system_text


def test_chat_voice_pass_scratchpad_carries_high_agreement_directive(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """End-to-end converse: when hemispheres agree, the voice pass
    scratchpad must include the high-band directive — confident
    register, no hedging."""
    # Identical responses → agreement 1.0 → terminate at pass 0.
    # +1 entry on left for the voice pass that follows.
    left_fake.responses = ["consensus reached", "voice"]
    right_fake.responses = ["consensus reached"]

    response = client.post("/v1/chat", json={"message": "hi"})
    assert response.status_code == 200, response.text

    voice_call = left_fake.calls[-1]
    system_msgs = [m for m in voice_call.messages if m.role.value == "system"]
    assert system_msgs
    system_text = system_msgs[0].content
    # High-band signal words.
    assert "conviction" in system_text.lower() or "certainty" in system_text.lower()


def test_chat_incognito_does_not_persist_to_memory(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """Incognito turns must leave no trace in memory.

    Regression guard for v0.2.1 incognito mode: validates that an
    incognito chat round-trip succeeds, returns a voice-pass reply, AND
    that the conversationId returned is NOT resolvable as a real
    memory conversation (a normal chat creates a memory-backed
    conversation; incognito does not)."""
    # Run an incognito turn. Voice pass is the only LLM call —
    # deliberation loop runs both hemispheres + voice on left.
    left_fake.responses = ["agree", "voice reply"]
    right_fake.responses = ["agree"]

    response = client.post(
        "/v1/chat",
        json={"message": "hello, stranger", "incognito": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["message"]["content"] == "voice reply"

    # The conversationId in the response must NOT exist in memory.
    # Non-incognito chat would have created it; incognito skips
    # memory.create() and synthesizes an ephemeral UUID instead.
    convo_id = body["conversationId"]
    fetched = client.get(f"/v1/conversations/{convo_id}")
    assert fetched.status_code == 404, (
        "incognito turn leaked into memory — found conversation "
        f"{convo_id} after a turn that should have left no trace"
    )


def test_chat_incognito_honors_request_history(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """When incognito is true, conversation history comes from the
    request body, not memory. The history field is the bridge that
    lets a multi-turn incognito conversation work without persistence."""
    left_fake.responses = ["follow-up agree", "voice follow-up"]
    right_fake.responses = ["follow-up agree"]

    # Second turn of an incognito conversation: the caller supplies the
    # prior turn pair via `history`.
    response = client.post(
        "/v1/chat",
        json={
            "message": "what did I just say?",
            "incognito": True,
            "history": [
                {"role": "user", "content": "I like Tom Hanks movies."},
                {"role": "assistant", "content": "Castaway is the right answer."},
            ],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["message"]["content"] == "voice follow-up"

    # Verify the hemispheres saw the supplied history. Both hemispheres
    # received the same history; check the left one.
    left_calls = left_fake.calls
    assert left_calls, "left hemisphere never called"
    # First pass should include the user-supplied history before the
    # current message.
    first_call_messages = left_calls[0].messages
    # Filter to user-role messages (excluding the leading system prompt).
    user_contents = [m.content for m in first_call_messages if m.role.value == "user"]
    assert "I like Tom Hanks movies." in user_contents
    assert "what did I just say?" in user_contents


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
