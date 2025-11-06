"""Tests for the Tessrax key vault service ensuring DLK compliance."""

from __future__ import annotations

from typing import List

import pytest
from fastapi.testclient import TestClient

from api import key_vault_service


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, List[dict]]:
    """Provide a FastAPI test client with ledger events captured in-memory."""

    events: List[dict] = []

    def _capture(event_type: str, payload: dict) -> None:
        events.append({"event_type": event_type, "payload": payload})

    monkeypatch.setattr(key_vault_service, "_append_ledger_event", _capture)
    return TestClient(key_vault_service.app), events


def test_health_and_signing(client: tuple[TestClient, List[dict]]) -> None:
    api_client, events = client
    health = api_client.get("/health")
    assert health.status_code == 200
    health_payload = health.json()
    assert health_payload["auditor"] == key_vault_service.AUDITOR
    assert len(health_payload["keys"]) == key_vault_service.TOTAL_KEYS

    message = "Governance layer test message"
    sign_primary = api_client.post("/sign", json={"message": message})
    assert sign_primary.status_code == 201
    primary_payload = sign_primary.json()
    assert primary_payload["key_id"] == key_vault_service.CURRENT_KEY_ID
    assert primary_payload["integrity_score"] >= 0.95

    sign_secondary = api_client.post(
        "/sign",
        json={"message": message, "key_id": "key2"},
    )
    assert sign_secondary.status_code == 201
    secondary_payload = sign_secondary.json()
    assert secondary_payload["key_id"] == "key2"

    verify_response = api_client.post(
        "/verify",
        json={
            "message": message,
            "signatures": [
                {
                    "signature": primary_payload["signature"],
                    "public_key": primary_payload["public_key"],
                    "key_id": primary_payload["key_id"],
                },
                {
                    "signature": secondary_payload["signature"],
                    "public_key": secondary_payload["public_key"],
                    "key_id": secondary_payload["key_id"],
                },
            ],
        },
    )
    assert verify_response.status_code == 200
    verify_payload = verify_response.json()
    assert verify_payload["quorum_met"] is True
    assert sorted(verify_payload["verified_signatures"]) == sorted(
        [primary_payload["key_id"], secondary_payload["key_id"]]
    )
    assert verify_payload["integrity_score"] >= 0.95
    assert events[-1]["event_type"] == "KEY_API_EVENT"
    assert events[-1]["payload"]["quorum_met"] is True


def test_quorum_threshold_enforced(client: tuple[TestClient, List[dict]]) -> None:
    api_client, _ = client
    message = "Threshold enforcement"
    sign_response = api_client.post("/sign", json={"message": message})
    assert sign_response.status_code == 201
    payload = sign_response.json()
    failure = api_client.post(
        "/verify",
        json={
            "message": message,
            "signatures": [
                {
                    "signature": payload["signature"],
                    "public_key": payload["public_key"],
                    "key_id": payload["key_id"],
                }
            ],
        },
    )
    assert failure.status_code == 400
    assert "At least" in failure.json()["detail"]
