"""Contract checks for the Tessrax Truth API service."""

import time

import jwt
import pytest

try:  # pragma: no cover - guard optional FastAPI dependency
    from fastapi.testclient import TestClient

    from tessrax_truth_api.main import app
    from tessrax_truth_api.utils import load_env
except ModuleNotFoundError:  # pragma: no cover - skip when FastAPI absent
    pytest.skip("fastapi not installed", allow_module_level=True)


TEST_SECRET = "test-jwt"  # must match middleware expectation


@pytest.fixture()
def auth_header(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Provide a deterministic bearer token for contract tests."""

    monkeypatch.setenv("JWT_SECRET", TEST_SECRET)
    load_env.cache_clear()
    now = int(time.time())
    payload = {
        "sub": "test",
        "tier": "free",
        "iat": now,
        "exp": now + 300,
        "iss": "tessrax-truth-api",
    }
    token = jwt.encode(payload, TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def test_detect_returns_receipt(auth_header: dict[str, str]) -> None:
    client = TestClient(app)
    response = client.post(
        "/detect",
        headers=auth_header,
        json={"claim_a": "A", "claim_b": "B", "tier": "free"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"verified", "unknown", "tampered"}
    assert "receipt_uuid" in payload
