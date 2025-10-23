from datetime import datetime

from tessrax.contradiction import ContradictionEngine
from tessrax.types import Claim


def make_claim(claim_id: str, value: float) -> Claim:
    return Claim(
        claim_id=claim_id,
        subject="Aurora Station",
        metric="emissions",
        value=value,
        unit="ktCO2e",
        timestamp=datetime.fromisoformat("2025-01-01T00:00:00"),
        source=claim_id,
    )


def test_detects_high_contradiction() -> None:
    claims = [make_claim("a", 100.0), make_claim("b", 60.0)]
    engine = ContradictionEngine(tolerance=0.1)

    contradictions = engine.detect(claims)

    assert len(contradictions) == 1
    record = contradictions[0]
    assert record.severity in {"high", "critical"}
    assert "Aurora Station emissions" in record.reasoning


def test_respects_tolerance() -> None:
    claims = [make_claim("a", 100.0), make_claim("b", 96.0)]
    engine = ContradictionEngine(tolerance=0.05)

    contradictions = engine.detect(claims)

    assert contradictions == []
