"""Deterministic adversarial sandbox for Tessrax governance testing.

This module satisfies AEP-001, RVC-001, and POST-AUDIT-001 by providing
cold-start deterministic simulations of ledger corruption. The
implementation avoids external side effects beyond governance receipts
and remains auditable under Tessrax Governance Kernel v16.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Dict, Iterable, List, Sequence, Tuple

from ..governance.receipts import write_receipt

_CANONICAL_BATCH_SIZE = 32


@dataclass
class ReceiptRecord:
    """Structured representation of a ledger receipt."""

    identifier: str
    payload: str
    ledger_hash: str
    canonical_hash: str

    def to_dict(self) -> Dict[str, str]:
        """Convert the record to a dictionary for downstream processing."""

        return {
            "id": self.identifier,
            "payload": self.payload,
            "ledger_hash": self.ledger_hash,
            "canonical_hash": self.canonical_hash,
        }


def _baseline_receipts(batch_size: int) -> List[ReceiptRecord]:
    """Create deterministic baseline receipts for simulation."""

    records: List[ReceiptRecord] = []
    for index in range(batch_size):
        payload = f"claim::{index:04d}::valid"
        canonical_hash = sha256(payload.encode("utf-8")).hexdigest()
        records.append(
            ReceiptRecord(
                identifier=f"R{index:04d}",
                payload=payload,
                ledger_hash=canonical_hash,
                canonical_hash=canonical_hash,
            )
        )
    return records


def inject_noise(receipts: Sequence[ReceiptRecord]) -> Tuple[List[ReceiptRecord], List[str]]:
    """Inject deterministic noise into provided receipts.

    The routine mutates copies of the provided records. Every fourth
    receipt receives a contradiction payload to emulate ledger
    corruption. We ensure reproducibility by relying solely on index
    arithmetic, avoiding pseudo-random generators.
    """

    corrupted: List[ReceiptRecord] = []
    tampered_ids: List[str] = []
    for index, record in enumerate(receipts):
        mutated = ReceiptRecord(
            identifier=record.identifier,
            payload=record.payload,
            ledger_hash=record.ledger_hash,
            canonical_hash=record.canonical_hash,
        )
        if index % 4 == 0:
            mutated.payload = f"{record.payload}::contradiction"
            tampered_ids.append(mutated.identifier)
        corrupted.append(mutated)
    return corrupted, tampered_ids


def simulate_attack(batch_size: int = _CANONICAL_BATCH_SIZE) -> Dict[str, Iterable[ReceiptRecord]]:
    """Simulate a deterministic adversarial batch of receipts."""

    baseline = _baseline_receipts(batch_size)
    corrupted, tampered_ids = inject_noise(baseline)
    return {
        "baseline": baseline,
        "corrupted": corrupted,
        "tampered_ids": tampered_ids,
    }


def _detect_corruption(receipt: ReceiptRecord) -> bool:
    """Return True when the receipt payload disagrees with its ledger hash."""

    payload_hash = sha256(receipt.payload.encode("utf-8")).hexdigest()
    return payload_hash != receipt.ledger_hash


def evaluate_recovery() -> Dict[str, float | List[Dict[str, str]]]:
    """Evaluate detection accuracy and produce recovery logs."""

    simulation = simulate_attack()
    corrupted = simulation["corrupted"]
    tampered_set = set(simulation["tampered_ids"])
    detections: List[str] = []
    recovery_log: List[Dict[str, str]] = []
    for record in corrupted:
        detected = _detect_corruption(record)
        if detected:
            detections.append(record.identifier)
            recovery_log.append(
                {
                    "id": record.identifier,
                    "recovered_hash": record.canonical_hash,
                    "status": "recovered",
                }
            )
    accuracy = 0.0
    if tampered_set:
        true_positive = sum(1 for identifier in detections if identifier in tampered_set)
        accuracy = true_positive / len(tampered_set)
    metrics = {
        "accuracy": round(accuracy, 3),
        "detections": len(detections),
        "tampered": len(tampered_set),
        "recovery_log": recovery_log,
    }
    integrity = 0.97 if accuracy >= 0.9 else 0.4
    write_receipt(
        "tessrax.core.sandbox.adversarial_sim",
        "verified",
        {k: v for k, v in metrics.items() if k != "recovery_log"},
        integrity,
    )
    return metrics


def _self_test() -> bool:
    """Run deterministic self-test with double-lock verification."""

    metrics = evaluate_recovery()
    assert metrics["accuracy"] >= 0.9, "Detection accuracy below threshold"
    assert metrics["detections"] >= metrics["tampered"], "Detection count mismatch"
    assert len(metrics["recovery_log"]) == metrics["detections"], "Recovery log mismatch"
    return True


if __name__ == "__main__":
    assert _self_test(), "Self-test failed"
