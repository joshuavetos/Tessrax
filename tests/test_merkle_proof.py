"""Tests for the governed Merkle proof generator and verifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from tessrax.core import merkle_proof
from tessrax.core.merkle_engine import MerkleEngine


@pytest.fixture()
def governed_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Path]:
    """Create a governed ledger with sample receipts for proof testing."""

    ledger_path = tmp_path / "ledger.jsonl"
    engine = MerkleEngine(ledger_path)
    sample = [
        {"directive": "alpha", "summary": "one"},
        {"directive": "beta", "summary": "two"},
        {"directive": "gamma", "summary": "three"},
    ]
    root = engine.build_and_store(sample)
    monkeypatch.setenv("TESSRAX_LEDGER_PATH", str(ledger_path))
    return root, ledger_path


@pytest.fixture()
def isolated_receipt_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect merkle proof receipts to a temporary location for tests."""

    receipt_path = tmp_path / "merkle_receipts.json"

    events: List[dict] = []
    monkeypatch.setattr(merkle_proof, "RECEIPT_LOG_PATH", receipt_path)
    monkeypatch.setattr(merkle_proof, "ledger_append", lambda payload: events.append(payload))
    # Ensure the new path is initialised and safepoint recorded
    merkle_proof._load_receipt_log()
    merkle_proof._record_safepoint_event()
    return receipt_path


def test_generate_and_verify_merkle_proof(
    governed_ledger: tuple[str, Path], isolated_receipt_log: Path
) -> None:
    root, ledger_path = governed_ledger
    engine = MerkleEngine(ledger_path)
    receipts = engine.load_receipts()
    target = receipts[1]
    bundle = merkle_proof.generate_proof(target["receipt_id"])
    assert bundle.merkle_root == root
    assert bundle.tree_size == len(receipts)
    assert merkle_proof.verify_proof(bundle, bundle.merkle_root)

    log_data = json.loads(isolated_receipt_log.read_text(encoding="utf-8"))
    assert log_data["events"], "Expected verification events to be recorded"
    assert log_data["events"][-1]["verified"] is True
    assert log_data["events"][-1]["receipt_id"] == target["receipt_id"]


def test_verify_rejects_invalid_root(
    governed_ledger: tuple[str, Path], isolated_receipt_log: Path
) -> None:
    root, ledger_path = governed_ledger
    engine = MerkleEngine(ledger_path)
    receipts = engine.load_receipts()
    bundle = merkle_proof.generate_proof(receipts[0]["receipt_id"])
    with pytest.raises(ValueError):
        merkle_proof.verify_proof(bundle, root + "deadbeef")
