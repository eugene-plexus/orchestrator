"""Identity-aware chat assembly.

Each chat turn the orchestrator pulls constitution + relevant self-model
entries + per-person relationship context, then builds *per-hemisphere*
system prompts so left and right each get a distinct preamble
identifying which side it is and what backend its twin is running.

These tests verify:

  - When identity is configured, both drivers receive system messages
    containing the constitution + person context + hemisphere preamble.
  - The two drivers' system prompts differ (each names itself and its twin).
  - `body.personId` is honored when supplied; otherwise the operator's
    personId is resolved from identity.
  - `body.systemPrompt` overrides the identity-assembled persona body
    but keeps the hemisphere preamble.
  - The v0.1 fallback path works unchanged when identity is unconfigured.
  - Identity-service outages degrade to the v0.1 path instead of
    failing the chat turn.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
from fastapi.testclient import TestClient

from eugene_plexus_orchestrator._generated.hemisphere_models import (
    GenerateRequest,
    Role,
)
from eugene_plexus_orchestrator._generated.models import (
    Constitution,
    Person,
    RelationshipSummary,
    SelfModelEntry,
)
from eugene_plexus_orchestrator.identity import InProcessIdentity
from eugene_plexus_orchestrator.settings import Settings
from tests.conftest import FakeHemisphereClient


def _system_message(req: GenerateRequest) -> str:
    """Pluck the system prompt out of a recorded GenerateRequest."""
    for msg in req.messages:
        if msg.role == Role.system:
            return msg.content
    return ""


def test_chat_falls_back_to_v01_path_when_identity_unconfigured(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """With identity off (the default in conftest), chat behaves like
    v0.1: a single shared persona body drawn from `defaultSystemPrompt`,
    just wrapped in the hemisphere preamble."""
    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    response = client.post("/v1/chat", json={"message": "hello"})
    assert response.status_code == 200, response.text

    left_sys = _system_message(left_fake.calls[0])
    right_sys = _system_message(right_fake.calls[0])
    # v0.2.x dropped the system-prompt preamble entirely. The persona
    # body IS the system prompt for both hemispheres. No architecture
    # meta, no per-side labels — divergence comes from the models
    # themselves and from per-pass cross-talk (the "you also
    # considered" message in pass 1+), not from preamble framing.
    assert "Eugene" in left_sys
    assert "Eugene" in right_sys
    # Both hemispheres see the same persona-only system prompt.
    assert left_sys == right_sys


def test_chat_with_identity_threads_constitution_into_both_hemispheres(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """When identity is wired in, both hemispheres' system prompts
    include constitution fields (name, pronouns, coreValues, freeText).
    The hemisphere preamble distinguishes the two."""
    from eugene_plexus_orchestrator.app import create_app
    from eugene_plexus_orchestrator.memory import InProcessMemory

    app = create_app(settings=settings)
    app.state.drivers = [left_fake, right_fake]
    app.state.memory = InProcessMemory()
    app.state.memory_url = "in-process"
    app.state.identity = InProcessIdentity(
        constitution=Constitution(
            name="Eugene",
            pronouns="they/them",
            coreValues=["honesty", "intellectual humility"],
            freeText="You like to cite analogies from neuroscience.",
        ),
    )
    app.state.identity_url = "in-process"

    with TestClient(app) as client:
        left_fake.responses = ["hi"]
        right_fake.responses = ["hi"]
        response = client.post("/v1/chat", json={"message": "hello"})
    assert response.status_code == 200, response.text

    for fake in (left_fake, right_fake):
        sys_msg = _system_message(fake.calls[0])
        assert "Your name: Eugene." in sys_msg
        assert "Pronouns: they/them." in sys_msg
        assert "honesty" in sys_msg
        assert "intellectual humility" in sys_msg
        assert "cite analogies from neuroscience" in sys_msg

    left_sys = _system_message(left_fake.calls[0])
    right_sys = _system_message(right_fake.calls[0])
    # v0.2.x dropped the system-prompt preamble; both hemispheres see
    # the same persona-only prompt. Per-hemisphere variation is a
    # planned v0.3 knob.
    assert left_sys == right_sys


def test_chat_uses_explicit_personId_relationship_when_supplied(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """A connector adapter supplies `personId` on the chat request.
    The orchestrator fetches that person's relationship context and
    injects it; the operator is NOT used as fallback."""
    from eugene_plexus_orchestrator.app import create_app
    from eugene_plexus_orchestrator.memory import InProcessMemory

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
        relationshipNote="my wife",
    )
    other_summary = RelationshipSummary(
        personId=other_id,
        lastUpdated=datetime.now(UTC),
        summary="Sarah and you chat several times a day, mostly casual.",
        turnCount=42,
    )

    app = create_app(settings=settings)
    app.state.drivers = [left_fake, right_fake]
    app.state.memory = InProcessMemory()
    app.state.memory_url = "in-process"
    app.state.identity = InProcessIdentity(
        persons=[operator, other],
        relationships={other_id: other_summary},
    )
    app.state.identity_url = "in-process"

    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={"message": "hello", "personId": str(other_id)},
        )
    assert response.status_code == 200, response.text

    sys_msg = _system_message(left_fake.calls[0])
    assert "Sarah" in sys_msg
    assert "my wife" in sys_msg
    assert "several times a day" in sys_msg
    # Operator's name does NOT leak into another-person's chat turn.
    assert "Troy" not in sys_msg


def test_chat_resolves_operator_personId_when_omitted(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """UI chat calls omit personId by design. Orchestrator looks up the
    operator's personId from identity and uses their relationship
    context (which for the operator is mostly the displayName + a
    "your operator" marker)."""
    from eugene_plexus_orchestrator.app import create_app
    from eugene_plexus_orchestrator.memory import InProcessMemory

    operator_id = uuid4()
    operator = Person(
        personId=operator_id,
        displayName="Troy",
        isOperator=True,
        createdAt=datetime.now(UTC),
    )

    app = create_app(settings=settings)
    app.state.drivers = [left_fake, right_fake]
    app.state.memory = InProcessMemory()
    app.state.memory_url = "in-process"
    app.state.identity = InProcessIdentity(persons=[operator])
    app.state.identity_url = "in-process"

    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    with TestClient(app) as client:
        response = client.post("/v1/chat", json={"message": "hello"})
    assert response.status_code == 200, response.text

    sys_msg = _system_message(left_fake.calls[0])
    assert "Troy" in sys_msg
    # The "(your operator…)" marker tells Eugene who they're talking to.
    assert "your operator" in sys_msg


def test_chat_filters_self_model_entries_by_related_person(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """Self-model entries scoped to other persons must not leak when
    talking to a different person."""
    from eugene_plexus_orchestrator.app import create_app
    from eugene_plexus_orchestrator.memory import InProcessMemory

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

    entry_about_operator = SelfModelEntry(
        id=uuid4(),
        topic="user-troy",
        content="With Troy you tend to switch into engineering mode.",
        createdAt=datetime.now(UTC),
        relatedPersonIds=[operator_id],
    )
    entry_about_other = SelfModelEntry(
        id=uuid4(),
        topic="user-sarah",
        content="With Sarah you tend to be more playful.",
        createdAt=datetime.now(UTC),
        relatedPersonIds=[other_id],
    )

    app = create_app(settings=settings)
    app.state.drivers = [left_fake, right_fake]
    app.state.memory = InProcessMemory()
    app.state.memory_url = "in-process"
    app.state.identity = InProcessIdentity(
        persons=[operator, other],
        self_model_entries=[entry_about_operator, entry_about_other],
    )
    app.state.identity_url = "in-process"

    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={"message": "hello", "personId": str(other_id)},
        )
    assert response.status_code == 200, response.text

    sys_msg = _system_message(left_fake.calls[0])
    assert "more playful" in sys_msg
    assert "engineering mode" not in sys_msg


def test_chat_systemPrompt_overrides_identity_persona_keeps_preamble(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """Caller-supplied `systemPrompt` is verbatim persona content. The
    hemisphere preamble (which-side-am-I) is still applied — that's a
    correctness property, not a persona choice."""
    from eugene_plexus_orchestrator.app import create_app
    from eugene_plexus_orchestrator.memory import InProcessMemory

    app = create_app(settings=settings)
    app.state.drivers = [left_fake, right_fake]
    app.state.memory = InProcessMemory()
    app.state.memory_url = "in-process"
    app.state.identity = InProcessIdentity(
        constitution=Constitution(name="Eugene", coreValues=["honesty"]),
    )
    app.state.identity_url = "in-process"

    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={"message": "hello", "systemPrompt": "TEMPORARILY ROLE: pirate"},
        )
    assert response.status_code == 200, response.text

    sys_msg = _system_message(left_fake.calls[0])
    # Identity body suppressed.
    assert "honesty" not in sys_msg
    assert "Your name: Eugene." not in sys_msg
    # Operator override is the entire system prompt — no preamble in
    # v0.2.x. The operator's override is what we send verbatim.
    assert sys_msg == "TEMPORARILY ROLE: pirate"


def test_chat_degrades_when_identity_service_is_unreachable(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """If identity is configured but unreachable mid-assembly, the chat
    must NOT fail — degrade to the defaultSystemPrompt path. Identity
    is enriching context, not load-bearing for the chat turn itself."""
    from eugene_plexus_orchestrator.app import create_app
    from eugene_plexus_orchestrator.memory import InProcessMemory

    class _FailingIdentity:
        async def get_constitution(self) -> Constitution:
            raise httpx.ConnectError("identity dead")

        async def query_self_model(
            self,
            *,
            topic: str | None = None,
            person_id: UUID | None = None,
            limit: int = 5,
        ) -> list[SelfModelEntry]:
            raise httpx.ConnectError("identity dead")

        async def get_relationship(self, person_id: UUID) -> RelationshipSummary | None:
            raise httpx.ConnectError("identity dead")

        async def list_persons(self) -> list[Person]:
            raise httpx.ConnectError("identity dead")

        async def aclose(self) -> None:
            return None

    app = create_app(settings=settings)
    app.state.drivers = [left_fake, right_fake]
    app.state.memory = InProcessMemory()
    app.state.memory_url = "in-process"
    app.state.identity = _FailingIdentity()
    app.state.identity_url = "http://127.0.0.1:8084"

    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    with TestClient(app) as client:
        response = client.post("/v1/chat", json={"message": "hello"})
    # Critically: 200, not 502. Identity outage is recoverable.
    assert response.status_code == 200, response.text

    sys_msg = _system_message(left_fake.calls[0])
    # Fallback persona came from defaultSystemPrompt (which mentions Eugene).
    assert "Eugene" in sys_msg


def test_pass0_preambles_are_identical_for_both_hemispheres(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """v0.2.x dropped the per-hemisphere preamble entirely. The system
    prompt is the persona body for both drivers — no architecture
    meta, no per-side labels, no "you are the left hemisphere" framing.

    Earlier versions added a distinguishing preamble ("you are the
    left hemisphere, your twin is the right hemisphere running on
    backend X") to make cross-vendor bicameral commitment visible.
    That backfired: the LLMs started addressing the orchestrator and
    treating each other as siblings to chat with. We now derive
    divergence from the underlying models being different (cross-
    vendor) and from per-pass cross-talk ("you also considered, from
    a slightly different angle..."), which surfaces the other side's
    content as the SAME Eugene's prior thought.

    This test pins the no-preamble invariant so a future change that
    re-introduces architectural framing gets caught with the why
    attached. Per-hemisphere persona variation is planned for v0.3
    (operator-selectable).
    """
    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    response = client.post("/v1/chat", json={"message": "hello"})
    assert response.status_code == 200, response.text

    left_sys = _system_message(left_fake.calls[0])
    right_sys = _system_message(right_fake.calls[0])
    assert left_sys == right_sys
    # No per-side labels surfaced into the prompt.
    assert "left hemisphere" not in left_sys
    assert "right hemisphere" not in left_sys
    # No architecture meta.
    assert "synthetic consciousness" not in left_sys
    assert "inner voice" not in left_sys
