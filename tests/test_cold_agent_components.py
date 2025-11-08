"""Unit tests for individual Cold Agent runtime components."""

from __future__ import annotations

from typing import Iterable, List

from tessrax.cold_agent.audit_capsule import AuditCapsule
from tessrax.cold_agent.contradiction_engine import ContradictionEngine, ContradictionRecord
from tessrax.cold_agent.integrity_ledger import IntegrityLedger
from tessrax.cold_agent.receipt_emitter import ReceiptEmitter
from tessrax.cold_agent.schema_validator import SchemaValidator
from tessrax.cold_agent.state_canonicalizer import StateCanonicalizer


def _serialize(records: Iterable[ContradictionRecord]) -> List[dict]:
    return [{"key": record.key, "old": record.old, "new": record.new} for record in records]


def test_component_round_trip_generates_auditable_receipt():
    """End-to-end component orchestration yields a verifiable receipt."""

    validator = SchemaValidator()
    validator.register_required_fields("weather", ["temperature"])

    canonicalizer = StateCanonicalizer()
    engine = ContradictionEngine()
    emitter = ReceiptEmitter()
    ledger = IntegrityLedger()
    audit = AuditCapsule()

    first_event = {"temperature": 70}
    second_event = {"temperature": 72}

    first_validation = validator.validate(first_event, "weather")
    assert first_validation.valid is True
    assert first_validation.errors == []

    first_apply = canonicalizer.apply(first_event)
    assert first_apply.diff == first_event
    assert first_apply.previous_state == {}

    second_validation = validator.validate(second_event, "weather")
    assert second_validation.valid is True

    second_apply = canonicalizer.apply(second_event)
    assert second_apply.previous_state == {"temperature": 70}
    assert second_apply.state == {"temperature": 72}

    contradiction_result = engine.detect(
        second_apply.previous_state,
        second_event,
        resulting_state=second_apply.state,
    )
    assert len(contradiction_result.contradictions) == 1
    record = contradiction_result.contradictions[0]
    assert record.key == "temperature"
    assert record.old == 70
    assert record.new == 72

    receipt = emitter.emit(
        event=second_event,
        pre_hash=second_apply.pre,
        post_hash=second_apply.post,
        contradictions=_serialize(contradiction_result.contradictions),
        metrics={"C_e": 1.0, "L_p": 1.0, "R_r": 1.0, "S_c": 1.0, "O_v": 1.0},
    )
    assert receipt["audit"]["auditor"] == "Tessrax Governance Kernel v16"
    assert receipt["pre_state_hash"] == second_apply.pre
    assert receipt["post_state_hash"] == second_apply.post

    ledger.append(receipt)
    root = ledger.merkle_root()
    assert ledger.audit_metadata()["entries"] == 1

    audit_result = audit.verify(ledger.replay(), root)
    assert audit_result["status"] is True
    assert audit_result["computed_root"] == root
    assert audit_result["expected_root"] == root
