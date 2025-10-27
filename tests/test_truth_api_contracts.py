"""Contract checks for the Tessrax Truth API service."""

import time

import jwt
import pytest

try:  # pragma: no cover - guard optional FastAPI dependency
    from fastapi.testclient import TestClient
    from tessrax_truth_api.main import app
except ModuleNotFoundError:  # pragma: no cover - skip when FastAPI absent
    pytest.skip("fastapi not installed", allow_module_level=True)


SECRET = "test-secret"


@pytest.fixture(scope="session")
def auth_header() -> dict[str, str]:
    """Provide a deterministic bearer token for contract tests."""

    payload = {"sub": "test", "iat": int(time.time())}
    token = jwt.encode(payload, SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def test_detect_returns_receipt(auth_header: dict[str, str]) -> None:
    client = TestClient(app)
    response = client.post(
        "/detect",
        headers=auth_header,
        json={"claim_a": "A", "claim_b": "B"},
    )

    assert response.status_code == 200
    assert "receipt" in response.json()
