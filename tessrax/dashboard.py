"""Dashboard helper utilities."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from tessrax.types import LedgerReceipt


def build_snapshot(receipts: Iterable[LedgerReceipt]) -> dict[str, object]:
    """Aggregate ledger receipts into dashboard-friendly metrics."""

    receipts = list(receipts)
    action_counts = Counter(receipt.decision.action for receipt in receipts)
    severity_counts = Counter(
        receipt.decision.contradiction.severity for receipt in receipts
    )
    clarity_total = sum(receipt.decision.clarity_fuel for receipt in receipts)

    return {
        "total_receipts": len(receipts),
        "actions": dict(action_counts),
        "severities": dict(severity_counts),
        "clarity_fuel_total": round(clarity_total, 3),
        "latest_event": (
            receipts[-1].decision.issued_at.isoformat() if receipts else None
        ),
    }
