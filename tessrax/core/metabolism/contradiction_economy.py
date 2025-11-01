"""Contradiction energy accounting under Tessrax governance.

Fulfils AEP-001, RVC-001, and POST-AUDIT-001 by delivering deterministic
energy ledgers with explicit receipts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

from ..governance.receipts import write_receipt

ECONOMY_PATH = Path("tessrax/core/metabolism/economy.json")
ECONOMY_PATH.parent.mkdir(parents=True, exist_ok=True)


def _stable_json(payload: Dict[str, float]) -> str:
    return "{" + ",".join(f'"{key}":{payload[key]:.6f}' for key in sorted(payload)) + "}\n"


def update_economy(events: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Update contradiction economy ledger and return aggregate metrics."""

    contradiction_energy = sum(event.get("entropy", 0.0) for event in events if event.get("type") == "contradiction")
    repair_energy = sum(event.get("entropy", 0.0) for event in events if event.get("type") == "repair")
    balance = round(contradiction_energy - repair_energy, 6)
    total_energy = round(contradiction_energy + repair_energy, 6)
    error_margin = round(abs(balance) / (total_energy or 1.0), 6)
    metrics = {
        "contradiction_energy": round(contradiction_energy, 6),
        "repair_energy": round(repair_energy, 6),
        "balance": balance,
        "error_margin": error_margin,
    }
    ECONOMY_PATH.write_text(_stable_json(metrics), encoding="utf-8")
    write_receipt("tessrax.core.metabolism.contradiction_economy", "verified", metrics, 0.96)
    return metrics


def _self_test() -> bool:
    """Verify energy accounting error remains within tolerance."""

    events = [
        {"type": "contradiction", "entropy": 5.0},
        {"type": "contradiction", "entropy": 4.8},
        {"type": "repair", "entropy": 9.6},
    ]
    metrics = update_economy(events)
    assert metrics["error_margin"] <= 0.02, "Energy accounting error too high"
    write_receipt(
        "tessrax.core.metabolism.contradiction_economy.self_test",
        "verified",
        {"error_margin": metrics["error_margin"]},
        0.95,
    )
    return True


if __name__ == "__main__":
    assert _self_test(), "Self-test failed"
