
"""
Metabolic reconciliation engine (Refactored for Stability and Weighted Governance).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, StatisticsError
from typing import Any, Dict, List, Optional

from tessrax.audit import AuditKernel
from tessrax.governance import router
from tessrax.ledger import Ledger

@dataclass
class Claim:
    agent_id: str
    value: Any
    truth_score: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class Dispute:
    dispute_id: str
    claims: List[Claim]
    context: Dict[str, Any] = field(default_factory=dict)

class ReconciliationEngine:
    def __init__(self, ledger: Optional[Ledger] = None, audit_kernel: Optional[AuditKernel] = None):
        """
        Initialize with optional dependency injection for easier testing.
        """
        self.ledger = ledger or Ledger()
        self.audit = audit_kernel or AuditKernel()
        self.logger = logging.getLogger("tessrax.metabolism")

    def reconcile_dispute(self, dispute: Dispute) -> Any:
        """
        Resolves conflicting claims using weighted averaging and truth scores.
        """
        if not dispute.claims:
            self.logger.warning(f"No claims found for dispute {dispute.dispute_id}")
            return None

        # Determine if we are dealing with numerical data for averaging
        numerical_claims = [c for c in dispute.claims if isinstance(c.value, (int, float))]

        if len(numerical_claims) == len(dispute.claims):
            return self._weighted_mean_resolution(numerical_claims)
        
        # Fallback to highest truth score for non-numerical or mixed data
        return self._highest_confidence_resolution(dispute.claims)

    def _weighted_mean_resolution(self, claims: List[Claim]) -> float:
        """Calculates a mean weighted by the truth_score of each agent."""
        try:
            total_weight = sum(c.truth_score for c in claims)
            if total_weight == 0:
                return mean(c.value for c in claims)
            
            weighted_sum = sum(c.value * c.truth_score for c in claims)
            return weighted_sum / total_weight
        except (StatisticsError, ZeroDivisionError):
            return 0.0

    def _highest_confidence_resolution(self, claims: List[Claim]) -> Any:
        """Returns the value from the agent with the highest truth score."""
        best_claim = max(claims, key=lambda c: c.truth_score)
        return best_claim.value

    def create_receipt(self, dispute_id: str, final_value: Any):
        """Records the final decision to the ledger with a cryptographic signature."""
        receipt = {
            "dispute_id": dispute_id,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "final_value": final_value,
            "status": "reconciled"
        }
        self.ledger.write_receipt(receipt)
        self.logger.info(f"Governance receipt issued for {dispute_id}")

# --- Keep existing CLI boilerplate if needed for your environment ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tessrax Reconciliation CLI")
    parser.add_name = "reconcile"
    # CLI logic would follow here...
