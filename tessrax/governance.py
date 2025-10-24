"""Governance kernel simulation."""

from __future__ import annotations

from .governance_security import DecisionSignature, SignatureAuthority
from .types import ContradictionRecord, GovernanceDecision


class GovernanceKernel:
    """Apply Tessrax protocols to contradiction records."""

    def __init__(
        self,
        memory_weight: float = 0.25,
        metabolism_weight: float = 0.25,
        governance_weight: float = 0.3,
        trust_weight: float = 0.2,
        *,
        signature_authority: SignatureAuthority | None = None,
    ) -> None:
        self.weights = {
            "memory": memory_weight,
            "metabolism": metabolism_weight,
            "governance": governance_weight,
            "trust": trust_weight,
        }
        self._signature_authority = signature_authority

    def process(self, contradiction: ContradictionRecord) -> GovernanceDecision:
        action = self._select_action(contradiction.severity)
        clarity = self._clarity_fuel(contradiction.delta)
        rationale = self._rationale(contradiction)
        decision = GovernanceDecision(
            contradiction=contradiction,
            action=action,
            clarity_fuel=clarity,
            rationale=rationale,
        )
        if self._signature_authority is not None:
            signature = self._signature_authority.sign(decision)
            self._attach_signature(decision, signature)
        return decision

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

    @staticmethod
    def _attach_signature(decision: GovernanceDecision, signature: DecisionSignature) -> None:
        decision.timestamp_token = signature.timestamp_token
        decision.signature = signature.signature
