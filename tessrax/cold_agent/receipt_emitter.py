"""Receipt emission for Cold Agent operations."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Dict, Iterable, List

AUDIT_METADATA = {
    "auditor": "Tessrax Governance Kernel v16",
    "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
}


class ReceiptEmitter:
    """Constructs deterministic receipts for ledger storage."""

    def emit(
        self,
        event: Dict[str, object],
        pre_hash: str,
        post_hash: str,
        contradictions: Iterable[Dict[str, object]],
        metrics: Dict[str, float],
    ) -> Dict[str, object]:
        """Emit a receipt capturing the provided execution context."""

        if not isinstance(event, dict):
            raise TypeError("event must be a dictionary")
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dictionary")
        if not isinstance(pre_hash, str) or not isinstance(post_hash, str):
            raise TypeError("Hashes must be hexadecimal strings")

        contradiction_list: List[Dict[str, object]] = []
        for record in contradictions:
            if not isinstance(record, dict):
                raise TypeError("contradictions must yield dictionaries")
            contradiction_list.append(dict(record))

        receipt_core = {
            "event": event,
            "timestamp": int(time.time()),
            "pre_state_hash": pre_hash,
            "post_state_hash": post_hash,
            "contradictions": contradiction_list,
            "metrics": metrics,
        }
        serialized = json.dumps(receipt_core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        runtime_signature = hashlib.sha256(serialized).hexdigest()
        receipt = {
            **receipt_core,
            "receipt_id": hashlib.sha256(uuid.uuid4().bytes).hexdigest(),
            "signature": runtime_signature,
            "audit": {**AUDIT_METADATA},
        }
        assert len(receipt["receipt_id"]) == 64, "Receipt identifier must be SHA-256 length"
        return receipt
