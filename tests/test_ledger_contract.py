from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tessrax.ledger import Ledger, compute_merkle_root
from tests.test_ledger import make_decision


def _manual_merkle(entries):
    nodes = [hashlib.sha256(json.dumps(entry, sort_keys=True).encode("utf-8")).hexdigest() for entry in entries]
    while len(nodes) > 1:
        next_level = []
        for index in range(0, len(nodes), 2):
            left = nodes[index]
            right = nodes[index + 1] if index + 1 < len(nodes) else nodes[index]
            next_level.append(hashlib.sha256((left + right).encode("utf-8")).hexdigest())
        nodes = next_level
    return nodes[0]


def test_merkle_root_matches_manual_computation():
    payloads = [
        {"decision_id": "d1", "severity": 0.1},
        {"decision_id": "d2", "severity": 0.2},
        {"decision_id": "d3", "severity": 0.3},
    ]
    expected = _manual_merkle(payloads)
    assert compute_merkle_root(payloads) == expected


def test_append_batch_records_merkle_root() -> None:
    ledger = Ledger()
    decisions = [make_decision() for _ in range(3)]
    root = ledger.append_batch(decisions)
    assert root is not None
    for receipt in ledger.receipts():
        assert receipt.sub_merkle_root == root


def test_signature_chain_offline(tmp_path: Path) -> None:
    ledger = Ledger()
    first = ledger.append(make_decision(), signature="sig-001")
    second = ledger.append(make_decision(), signature="sig-002")
    export_path = tmp_path / "contract-ledger.jsonl"
    ledger.export(export_path)

    on_disk = [json.loads(line) for line in export_path.read_text(encoding="utf-8").splitlines()]
    assert all(entry.get("signature") for entry in on_disk)

    prev_hash = "GENESIS"
    required_fields = {
        "event_type",
        "timestamp",
        "decision_id",
        "action",
        "severity",
        "clarity_fuel",
        "subject",
        "metric",
        "rationale",
        "protocol",
        "timestamp_token",
    }
    for entry in on_disk:
        summary = {key: entry[key] for key in required_fields if key in entry}
        digest = Ledger._hash_payload(prev_hash, summary)
        assert digest == entry["hash"]
        prev_hash = entry["hash"]

    assert first.signature == "sig-001"
    assert second.prev_hash == first.hash
