"""Tests for admin endpoints and the config protocol."""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from tests.conftest import FakeHemisphereClient


def test_admin_hemispheres_reports_pair(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    response = client.get("/v1/admin/hemispheres")
    assert response.status_code == 200
    body = response.json()
    assert body["left"]["reachable"] is True
    assert body["left"]["backend"] == "claude_code_cli"
    assert body["right"]["reachable"] is True
    assert body["right"]["backend"] == "codex_cli"


def test_admin_hemispheres_reports_individual_unreachability(
    client: TestClient,
    left_fake: FakeHemisphereClient,
) -> None:
    left_fake.info_error = httpx.ConnectError("connection refused")
    response = client.get("/v1/admin/hemispheres")
    assert response.status_code == 200
    body = response.json()
    assert body["left"]["reachable"] is False
    assert "connection refused" in body["left"]["error"]
    assert body["right"]["reachable"] is True


def test_admin_hemispheres_503_when_both_down(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    left_fake.info_error = httpx.ConnectError("dead")
    right_fake.info_error = httpx.ConnectError("dead")
    response = client.get("/v1/admin/hemispheres")
    assert response.status_code == 503


def test_admin_nt_state_returns_neutral_baseline(client: TestClient) -> None:
    response = client.get("/v1/admin/nt-state")
    assert response.status_code == 200
    body = response.json()
    for key in ("serotonin", "dopamine", "norepinephrine", "acetylcholine", "gaba", "glutamate"):
        assert body[key] == 0.5


def test_config_schema_lists_orchestrator_fields(client: TestClient) -> None:
    response = client.get("/v1/config/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["component"] == "orchestrator"
    keys = {f["key"] for f in body["fields"]}
    expected = {
        "leftDriverUrl",
        "rightDriverUrl",
        "port",
        "logLevel",
        "defaultMaxPasses",
        "agreementThreshold",
        "defaultSystemPrompt",
        "requestTimeoutSeconds",
    }
    assert expected.issubset(keys)


def test_config_get_then_patch_round_trip(client: TestClient) -> None:
    initial = client.get("/v1/config").json()
    assert initial["agreementThreshold"] == 0.5

    patch = client.patch(
        "/v1/config",
        json={"agreementThreshold": 0.75, "defaultMaxPasses": 5, "bogus": True},
    )
    assert patch.status_code == 200
    body = patch.json()
    assert "agreementThreshold" in body["applied"]
    assert "defaultMaxPasses" in body["applied"]
    rejected_keys = {r["key"] for r in body["rejected"]}
    assert "bogus" in rejected_keys
    # Both applied fields are read live at request time — no restart needed.
    assert body["requiresRestart"] is False

    follow = client.get("/v1/config").json()
    assert follow["agreementThreshold"] == 0.75
    assert follow["defaultMaxPasses"] == 5
