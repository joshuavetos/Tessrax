"""Contract checks for the Tessrax Truth API service."""

import pytest

try:  # pragma: no cover - guard optional FastAPI dependency
    from fastapi.testclient import TestClient
    from tessrax_truth_api.main import app
except ModuleNotFoundError:  # pragma: no cover - skip when FastAPI absent
    pytest.skip("fastapi not installed", allow_module_level=True)


def test_detect_returns_receipt() -> None:
    client = TestClient(app)
    response = client.post("/detect", json={"claim_a": "A", "claim_b": "B"})

    assert response.status_code == 200
    assert "receipt" in response.json()
