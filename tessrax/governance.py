"""Governance kernel simulation."""

from __future__ import annotations

from .types import ContradictionRecord, GovernanceDecision


class GovernanceKernel:
    """Apply Tessrax protocols to contradiction records."""

    def __init__(
        self,
        memory_weight: float = 0.25,
        metabolism_weight: float = 0.25,
        governance_weight: float = 0.3,
        trust_weight: float = 0.2,
    ) -> None:
        self.weights = {
            "memory": memory_weight,
            "metabolism": metabolism_weight,
            "governance": governance_weight,
            "trust": trust_weight,
        }

    def process(self, contradiction: ContradictionRecord) -> GovernanceDecision:
        action = self._select_action(contradiction.severity)
        clarity = self._clarity_fuel(contradiction.delta)
        rationale = self._rationale(contradiction)
        return GovernanceDecision(
            contradiction=contradiction,
            action=action,
            clarity_fuel=clarity,
            rationale=rationale,
        )

    def _select_action(self, severity: str) -> str:
        mapping = {
            "low": "ACKNOWLEDGE",
            "medium": "REMEDIATE",
            "high": "ESCALATE",
            "critical": "RESET",
        }
        return mapping.get(severity, "REVIEW")

    def _clarity_fuel(self, delta: float) -> float:
        detachment_score = min(max(delta * 4, 0.0), 2.0)
        return 12 * (detachment_score ** 1.5)

    def _rationale(self, contradiction: ContradictionRecord) -> str:
        protocol_summary = (
            f"Memory captured conflicting claims {contradiction.claim_a.claim_id} and {contradiction.claim_b.claim_id}. "
            f"Metabolism measured delta {contradiction.delta:.1%}. "
            "Governance applied quorum thresholds while Trust signalled observers."
        )
        return protocol_summary
