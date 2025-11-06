"""Tests for the Tessrax External Verifier Service API."""
from __future__ import annotations

import base64

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from verifier.service import app


client = TestClient(app)


def test_verifier_endpoint():
    """Ensure the /verify endpoint responds with a validation status or error."""
    sample = {"directive": "SAFEPOINT_TEST", "hash": "abc123"}
    body = {
        "payload": sample,
        "signature": base64.b64encode(b"fake").decode(),
        "public_key": base64.b64encode(b"A" * 32).decode(),
    }
    response = client.post("/verify", json=body)
    assert response.status_code in (200, 400)
    if response.status_code == 200:
        data = response.json()
        assert "audit_receipt" in data
        assert data["audit_receipt"]["status"] in {"PASS", "FAIL"}
