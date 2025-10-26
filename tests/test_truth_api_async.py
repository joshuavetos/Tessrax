from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

from tessrax_truth_api.main import create_app
from tessrax_truth_api.utils import issue_jwt, load_env


@pytest.fixture(autouse=True)
def override_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("TRUTH_API_LEDGER_PATH", str(ledger_path))
    monkeypatch.setenv("HMAC_SECRET", "test-hmac")
    monkeypatch.setenv("JWT_SECRET", "test-jwt")
    # reset cached env lookups so the application sees the override
    load_env.cache_clear()
    yield
    if ledger_path.exists():
        ledger_path.unlink()


pytestmark = pytest.mark.anyio("asyncio")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    test_app = create_app()
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client


def _auth_headers(tier: str) -> dict[str, str]:
    token = issue_jwt(tier, subject="test-client", minutes=5)
    return {"Authorization": f"Bearer {token}"}


async def test_onboard_endpoint_returns_token(client: AsyncClient):
    response = await client.post("/onboard", params={"key": "create", "tier": "pro"})
    assert response.status_code == status.HTTP_200_OK
    payload = response.json()
    assert payload["tier"] == "pro"
    assert isinstance(payload["token"], str)


async def test_detect_endpoint_generates_receipt(client: AsyncClient):
    headers = _auth_headers("free")
    response = await client.post(
        "/detect",
        json={"claim_a": "Cats are animals", "claim_b": "Cats are not animals", "tier": "free"},
        headers=headers,
    )
    assert response.status_code == status.HTTP_200_OK
    payload = response.json()
    assert payload["status"] == "verified"
    assert payload["verdict"] in {"contradiction", "aligned", "unknown"}
    assert "receipt_uuid" in payload


async def test_verify_receipt_returns_merkle_details(client: AsyncClient):
    headers = _auth_headers("free")
    detect_response = await client.post(
        "/detect",
        json={"claim_a": "Paris is in France", "claim_b": "Paris is not in France", "tier": "free"},
        headers=headers,
    )
    receipt_uuid = detect_response.json()["receipt_uuid"]

    verify_response = await client.get(f"/verify_receipt/{receipt_uuid}", headers=headers)
    assert verify_response.status_code == status.HTTP_200_OK
    payload = verify_response.json()
    assert payload["uuid"] == receipt_uuid
    assert payload["signature_valid"] is True


async def test_metrics_endpoint(client: AsyncClient):
    response = await client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    assert "truth_api_integrity" in response.text


async def test_self_test_reports_results(client: AsyncClient):
    response = await client.get("/self_test")
    assert response.status_code == status.HTTP_200_OK
    payload = response.json()
    assert len(payload["results"]) == 3
    statuses = {result["status"] for result in payload["results"]}
    assert {"verified", "unknown", "tampered"}.issubset(statuses)
