"""End-to-end NT integration tests through the chat handler.

Verify:
  - NT state evolves across turns (not reset on every call).
  - ChatResponse carries both `ntStateAtStart` and `ntStateAtEnd`.
  - High-cortisol NT bumps `max_passes` for the next turn.
  - GET /v1/admin/nt-state surfaces the live state, not a fresh neutral.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from eugene_plexus_orchestrator._generated.models import NTLevel
from eugene_plexus_orchestrator.bicameral.nt import neutral_state
from eugene_plexus_orchestrator.settings import Settings
from tests.conftest import FakeHemisphereClient


def test_chat_response_carries_start_and_end_nt_state(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """Both NTState snapshots ride on the response so the UI can render
    Eugene's cognitive arc per turn."""
    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    response = client.post("/v1/chat", json={"message": "hello"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert "ntStateAtStart" in body
    assert "ntStateAtEnd" in body
    # Quick single-pass convergence → GABA went up, dopamine went up.
    assert body["ntStateAtEnd"]["gaba"]["level"] > 0.5
    assert body["ntStateAtEnd"]["dopamine"]["level"] > 0.5


def test_nt_state_evolves_across_turns(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """Two consecutive turns: the second turn's `ntStateAtStart` should
    equal the first turn's `ntStateAtEnd` (modulo a tiny decay over the
    sub-millisecond gap between requests — assert convergence direction
    instead of exact equality)."""
    left_fake.responses = ["hi", "hi"]
    right_fake.responses = ["hi", "hi"]

    first = client.post("/v1/chat", json={"message": "first"}).json()
    second = client.post("/v1/chat", json={"message": "second"}).json()

    first_end_dopamine = first["ntStateAtEnd"]["dopamine"]["level"]
    second_start_dopamine = second["ntStateAtStart"]["dopamine"]["level"]
    # The second turn started with approximately where the first turn left off.
    assert abs(second_start_dopamine - first_end_dopamine) < 0.05


def test_admin_nt_state_returns_live_evolved_state(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """After a chat turn nudges dopamine up, GET /v1/admin/nt-state
    must reflect it — not return a fresh neutral every call."""
    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    client.post("/v1/chat", json={"message": "hello"})

    response = client.get("/v1/admin/nt-state")
    assert response.status_code == 200
    body = response.json()
    assert body["dopamine"]["level"] > 0.5


def test_high_cortisol_widens_modulated_max_passes(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """When the orchestrator starts with cortisol cranked, the chat
    handler's modulated max_passes should be higher than the configured
    base. This proves the modulation feeds the bicameral loop, not just
    the response payload."""
    from eugene_plexus_orchestrator.app import create_app
    from eugene_plexus_orchestrator.memory import InProcessMemory

    app = create_app(settings=settings)
    app.state.drivers = [left_fake, right_fake]
    app.state.memory = InProcessMemory()
    app.state.memory_url = "in-process"
    app.state.identity = None
    app.state.identity_url = ""

    # Pre-set NT state: cortisol pegged high → +2 passes; NE high → +1 pass.
    base = neutral_state()
    app.state.nt_state = base.model_copy(
        update={
            "cortisol": NTLevel(level=1.0, baseline=0.5, decay=base.cortisol.decay),
            "norepinephrine": NTLevel(
                level=1.0, baseline=0.5, decay=base.norepinephrine.decay
            ),
        }
    )

    # Always disagree so the loop runs to its max. Provide 6 responses
    # so the test doesn't run out mid-pass: configured base is 3, NT
    # modulation pushes it to 3 + 2 + 1 = 6.
    disagreements_l = [f"alpha-{i}" for i in range(6)]
    disagreements_r = [f"omega-{i}" for i in range(6)]
    left_fake.responses = disagreements_l
    right_fake.responses = disagreements_r

    with TestClient(app) as client:
        response = client.post("/v1/chat", json={"message": "test"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["passes"]) == 6  # base 3 + cortisol 2 + ne 1


def test_per_pass_latency_propagates_to_nt_observations(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """Each pass's max-of-hemispheres latency rides on the bicameral
    outcome and feeds NT.norepinephrine. We don't have a way to inject
    fake latencies into FakeHemisphereClient (it always reports 1ms);
    this test is a smoke check that the pipeline produces a sane
    ntStateAtEnd rather than crashing on the latency-aggregation path."""
    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    response = client.post("/v1/chat", json={"message": "hello"})
    assert response.status_code == 200, response.text
    body = response.json()
    # Very low latency (1ms) → NE pulled down from 0.5 by the
    # short-latency clause; assert direction, not magnitude.
    assert body["ntStateAtEnd"]["norepinephrine"]["level"] < 0.5
