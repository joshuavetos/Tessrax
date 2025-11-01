from datetime import datetime
from pathlib import Path

import pytest

from tessrax.ledger import GENESIS_HASH, Ledger, verify_file
from tessrax.types import Claim, ContradictionRecord, GovernanceDecision


def make_decision(action: str = "REMEDIATE") -> GovernanceDecision:
    claim_a = Claim(
        claim_id="a",
        subject="Atlas City",
        metric="water_quality",
        value=0.9,
        unit="index",
        timestamp=datetime.fromisoformat("2025-01-01T00:00:00"),
        source="source-a",
    )
    claim_b = Claim(
        claim_id="b",
        subject="Atlas City",
        metric="water_quality",
        value=0.7,
        unit="index",
        timestamp=datetime.fromisoformat("2025-01-02T00:00:00"),
        source="source-b",
    )
    contradiction = ContradictionRecord(
        claim_a=claim_a,
        claim_b=claim_b,
        severity="high",
        delta=0.25,
        reasoning="test",
    )
    return GovernanceDecision(
        contradiction=contradiction,
        action=action,
        clarity_fuel=12.0,
        rationale="unit-test",
    )


def test_append_sets_genesis_hash() -> None:
    ledger = Ledger()
    receipt = ledger.append(make_decision())

    assert receipt.prev_hash == GENESIS_HASH
    assert len(receipt.hash) == 64


def test_verify_round_trip(tmp_path: Path) -> None:
    ledger = Ledger()
    ledger.append(make_decision())
    ledger.append(make_decision(action="ESCALATE"))

    path = tmp_path / "ledger.jsonl"
    ledger.export(path)

    assert verify_file(path)


def test_verify_detects_tampering(tmp_path: Path) -> None:
    ledger = Ledger()
    ledger.append(make_decision())
    path = tmp_path / "ledger.jsonl"
    ledger.export(path)

    lines = path.read_text().splitlines()
    corrupt = (
        lines[0].replace("ESCALATE", "RESET")
        if "ESCALATE" in lines[0]
        else lines[0].replace("12.0", "24.0")
    )
    path.write_text("\n".join([corrupt]))

    with pytest.raises(ValueError):
        verify_file(path)
