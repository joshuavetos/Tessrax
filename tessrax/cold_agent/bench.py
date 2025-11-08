"""Executable benchmark wiring the Cold Agent runtime components together."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from tessrax.cold_agent.audit_capsule import AuditCapsule
from tessrax.cold_agent.contradiction_engine import ContradictionEngine
from tessrax.cold_agent.receipt_emitter import ReceiptEmitter
from tessrax.cold_agent.schema_validator import SchemaValidator
from tessrax.cold_agent.state_canonicalizer import StateCanonicalizer
from tessrax.cold_agent.integrity_ledger import IntegrityLedger

AUDIT_METADATA = {
    "auditor": "Tessrax Governance Kernel v16",
    "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
}

OUTPUT_PATH = Path("out/cold_agent_run_receipt.json")


def _default_dataset() -> Sequence[Dict[str, int]]:
    return (
        {"temperature": 70},
        {"temperature": 75},
        {"humidity": 40},
        {"humidity": 35},
    )


def _serialize_contradictions(records: Iterable[object]) -> List[Dict[str, object]]:
    payload: List[Dict[str, object]] = []
    for record in records:
        if hasattr(record, "key") and hasattr(record, "old") and hasattr(record, "new"):
            payload.append({"key": record.key, "old": record.old, "new": record.new})
        else:  # pragma: no cover - defensive guard
            raise TypeError("Contradiction records must expose key, old, and new attributes")
    return payload


def main(dataset: Sequence[Dict[str, object]] | None = None) -> Dict[str, object]:
    """Run the Cold Agent reference pipeline and persist an execution receipt."""

    dataset = dataset or _default_dataset()
    validator = SchemaValidator()
    engine = ContradictionEngine()
    canonicalizer = StateCanonicalizer()
    emitter = ReceiptEmitter()
    ledger = IntegrityLedger()
    audit = AuditCapsule()

    start = time.time()
    for index, event in enumerate(dataset):
        validation = validator.validate(event)
        if not validation.valid:
            raise ValueError(f"Dataset event failed validation: {validation.errors}")

        canonical_result = canonicalizer.apply(event)
        contradiction_result = engine.detect(
            canonical_result.previous_state,
            event,
            resulting_state=canonical_result.state,
        )

        contradiction_payload = _serialize_contradictions(contradiction_result.contradictions)
        metrics = {
            "C_e": 1.0,
            "L_p": float(len(contradiction_payload)),
            "R_r": 1.0,
            "S_c": 1.0,
            "O_v": 1.0,
            "index": float(index),
        }
        receipt = emitter.emit(
            event=event,
            pre_hash=canonical_result.pre,
            post_hash=canonical_result.post,
            contradictions=contradiction_payload,
            metrics=metrics,
        )
        ledger.append(receipt)

    root = ledger.merkle_root()
    audit_result = audit.verify(ledger.replay(), root)
    end = time.time()

    status = "PASS" if audit_result["status"] else "FAIL"
    integrity_score = 1.0 if status == "PASS" else 0.0
    runtime_ms = (end - start) * 1000
    summary = {
        "timestamp": int(time.time()),
        "runtime_info": {
            "dataset_size": len(dataset),
            "run_time_ms": round(runtime_ms, 3),
            "merkle_root": root,
        },
        "integrity_score": integrity_score,
        "status": status,
        "signature": _signature_payload(root, runtime_ms, len(dataset)),
        "auditor": AUDIT_METADATA["auditor"],
        "clauses": AUDIT_METADATA["clauses"],
        "ledger_metadata": ledger.audit_metadata(),
        "audit_result": audit_result,
    }

    if status != "PASS":  # pragma: no cover - defensive guard
        raise RuntimeError("Audit failed; refusing to persist non-compliant receipt")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return summary


def _signature_payload(root: str | None, runtime_ms: float, dataset_size: int) -> str:
    payload = json.dumps(
        {
            "root": root,
            "runtime_ms": round(runtime_ms, 3),
            "dataset_size": dataset_size,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return __import__("hashlib").sha256(payload).hexdigest()


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    receipt = main()
    print(json.dumps(receipt, indent=2, sort_keys=True))
