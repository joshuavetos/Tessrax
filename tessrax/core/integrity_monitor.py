"""Tessrax Continuous Integrity Monitor (v18.2).

Evaluates ledger health metrics and triggers key rotation when entropy or
drift thresholds are exceeded.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tessrax.core.key_vault import KeyVault
from tessrax.core.ledger import Ledger


class IntegrityMonitor:
    """Continuously evaluates ledger metrics and records governance receipts."""

    def __init__(
        self,
        ledger_path: str | Path = ".ledger.jsonl",
        *,
        threshold_entropy: float = 0.7,
        threshold_drift: float = 0.1,
    ) -> None:
        self.ledger = Ledger(ledger_path)
        self.key_vault = KeyVault(self.ledger)
        self.threshold_entropy = threshold_entropy
        self.threshold_drift = threshold_drift
        self._prev_count = 0

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------
    def compute_entropy(self) -> float:
        receipts = self.ledger.receipts()
        if not receipts:
            return 0.0
        tokens = [entry.get("hash", "") for entry in receipts if entry.get("hash")]
        if not tokens:
            return 0.0
        first_chars = [token[0] for token in tokens]
        total = len(first_chars)
        frequencies: dict[str, float] = {}
        for char in first_chars:
            frequencies[char] = frequencies.get(char, 0) + 1
        entropy = 0.0
        for value in frequencies.values():
            probability = value / total
            entropy -= probability * math.log(probability, 2)
        return entropy

    def compute_drift(self) -> float:
        count = len(self.ledger.receipts())
        if count == 0:
            self._prev_count = 0
            return 0.0
        drift = abs(count - self._prev_count) / max(count, 1)
        self._prev_count = count
        return drift

    # ------------------------------------------------------------------
    # Monitoring loop
    # ------------------------------------------------------------------
    async def monitor(self, interval: float = 10.0) -> None:
        while True:
            entropy = self.compute_entropy()
            drift = self.compute_drift()
            status: dict[str, Any] = {
                "directive": "INTEGRITY_STATUS",
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "entropy": entropy,
                "drift": drift,
                "auditor": "Tessrax Governance Kernel v16",
                "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "DLK-001"],
            }
            if entropy > self.threshold_entropy or drift > self.threshold_drift:
                rotation_receipt = self.key_vault.rotate_key()
                status["action"] = "KEY_ROTATION_TRIGGERED"
                status["rotation_receipt"] = rotation_receipt
            else:
                status["action"] = "NO_ACTION"
            self.ledger.append(status)
            await asyncio.sleep(interval)


__all__ = ["IntegrityMonitor"]
