from __future__ import annotations

from datetime import datetime

from tessrax.contradiction import ContradictionEngine
from tessrax.governance import GovernanceKernel
from tessrax.ledger import Ledger
from tessrax.types import Claim


def make_claim(claim_id: str, value: float) -> Claim:
    return Claim(
        claim_id=claim_id,
        subject="Helios", 
        metric="emissions",
        value=value,
        unit="ktCO2e",
        timestamp=datetime.fromisoformat("2025-01-01T00:00:00"),
        source=f"sensor-{claim_id}",
    )


def test_full_governance_cycle(tmp_path) -> None:
    claims = [make_claim("a", 100.0), make_claim("b", 40.0)]

    contradictions = ContradictionEngine(tolerance=0.05).detect(claims)
    assert contradictions, "Expected contradictions to trigger governance"

    kernel = GovernanceKernel()
    decision = kernel.process(contradictions[0])
    ledger = Ledger(path=tmp_path / "cycle.jsonl")

    receipt = ledger.append(decision, signature="cycle-signature")
    assert receipt.signature == "cycle-signature"
    assert ledger.verify()

    export = tmp_path / "cycle.jsonl"
    assert export.exists()
