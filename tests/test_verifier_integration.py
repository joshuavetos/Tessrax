"""Tests for the Tessrax external verifier integration (v18.7)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import federation.demo_federation as demo


class _MockResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, object]:
        return self._payload


@pytest.fixture(name="mock_verifier_paths")
def _mock_verifier_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    output_path = tmp_path / "verifier.json"
    receipt_path = tmp_path / "receipt.json"
    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(demo, "VERIFIER_OUTPUT", output_path)
    monkeypatch.setattr(demo, "VERIFIER_RECEIPT", receipt_path)
    monkeypatch.setattr(demo, "VERIFIER_LEDGER", ledger_path)
    return {
        "output": output_path,
        "receipt": receipt_path,
        "ledger": ledger_path,
    }


def test_external_verifier_integration(monkeypatch: pytest.MonkeyPatch, mock_verifier_paths: dict[str, Path]) -> None:
    verification_payload = {
        "nodes": {
            "alpha": {"verified": True},
            "beta": {"verified": True},
            "gamma": {"verified": True},
        },
        "verified": True,
    }

    def _mock_post(url: str, json: dict[str, object], timeout: int) -> _MockResponse:
        assert url == demo.VERIFIER_URL
        assert "roots" in json and "ledger_hashes" in json
        return _MockResponse(verification_payload)

    monkeypatch.setattr(demo.requests, "post", _mock_post)

    consensus = asyncio.run(demo.federated_run())

    assert consensus is True
    verifier_data = json.loads(mock_verifier_paths["output"].read_text())
    assert all(details["verified"] is True for details in verifier_data["nodes"].values())
    receipt_data = json.loads(mock_verifier_paths["receipt"].read_text())
    assert receipt_data["status"] == "PASS"
    assert receipt_data["safepoint"] == demo.SAFEPOINT_VERIFIER
    ledger_lines = [line for line in mock_verifier_paths["ledger"].read_text().splitlines() if line.strip()]
    assert ledger_lines, "Ledger should contain at least one entry"
