from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tessrax.audit import AuditKernel
from tessrax.governance import classify_contradiction, route_to_governance_lane
from tessrax.ledger import Ledger, compute_merkle_root
from tessrax.metabolism.reconcile import DriftTracker, ReconciliationEngine
from tessrax.trust import BayesianTrust
from tessrax.types import Claim, ContradictionRecord, GovernanceDecision


@pytest.fixture()
def sample_claims() -> tuple[Claim, Claim]:
    now = datetime.now(timezone.utc)
    claim_a = Claim(
        claim_id="c1",
        subject="domain",
        metric="uptime",
        value=0.9,
        unit="ratio",
        timestamp=now,
        source="sensor-a",
    )
    claim_b = Claim(
        claim_id="c2",
        subject="domain",
        metric="uptime",
        value=0.5,
        unit="ratio",
        timestamp=now,
        source="sensor-b",
    )
    return claim_a, claim_b


def test_classify_contradiction_heuristic(monkeypatch):
    monkeypatch.setattr("tessrax.governance._MODEL", None)
    result = classify_contradiction("alpha beta", "gamma delta")
    assert result == "semantic"


def test_route_to_governance_lane_high_priority(sample_claims):
    record = ContradictionRecord(
        claim_a=sample_claims[0],
        claim_b=sample_claims[1],
        severity="high",
        delta=0.4,
        reasoning="disagreement",
        confidence=0.92,
        contradiction_type="semantic",
    )
    assert route_to_governance_lane(record) == "high_priority_lane"


def test_drift_tracker_detects_change():
    tracker = DriftTracker()
    tracker.update(0.8, 1.0)
    tracker.update(0.6, 0.5)
    assert pytest.approx(tracker.drift(), rel=1e-6) == 0.5


def test_reconciliation_engine_records_drift(sample_claims):
    engine = ReconciliationEngine(AuditKernel(), ledger=Ledger())
    record = ContradictionRecord(
        claim_a=sample_claims[0],
        claim_b=sample_claims[1],
        severity="high",
        delta=0.3,
        reasoning="variance detected",
        confidence=0.8,
        contradiction_type="semantic",
    )
    statements = engine.reconcile([record])
    assert len(statements) == 1
    statement = statements[0]
    assert statement.clarity_fuel > 0
    meta_events = engine.ledger.meta_events()
    assert meta_events and meta_events[0]["governance_lane"] == "high_priority_lane"


def test_compute_merkle_root_and_batch(sample_claims):
    ledger = Ledger()
    record = ContradictionRecord(
        claim_a=sample_claims[0],
        claim_b=sample_claims[1],
        severity="medium",
        delta=0.3,
        reasoning="variance detected",
    )
    decision = GovernanceDecision(
        contradiction=record,
        action="ACKNOWLEDGE",
        clarity_fuel=1.0,
        rationale="test",
    )
    root = ledger.append_batch([decision])
    assert root
    receipts = ledger.receipts()
    assert receipts[0].sub_merkle_root == root
    payload = [decision.to_summary()]
    assert compute_merkle_root(payload) == root


def test_bayesian_trust_progression():
    trust = BayesianTrust()
    baseline = trust.score
    trust.update(success=True)
    assert trust.score > baseline
    trust.redeem()
    assert trust.score >= baseline
    serialised = trust.to_dict()
    assert set(serialised).issuperset({"alpha", "beta", "trust_score", "timestamp"})
