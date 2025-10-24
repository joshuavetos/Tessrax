"""Governance kernel simulation."""

from __future__ import annotations

from typing import Iterable, Mapping

from tessrax.governance_security import DecisionSignature, SignatureAuthority
from tessrax.meta_integrity.analytics import compute_epistemic_metrics
from tessrax.types import ContradictionRecord, GovernanceDecision


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
        epistemic_ledger: object | None = None,
    ) -> None:
        self.weights = {
            "memory": memory_weight,
            "metabolism": metabolism_weight,
            "governance": governance_weight,
            "trust": trust_weight,
        }
        self._signature_authority = signature_authority
        self._epistemic_ledger = epistemic_ledger
        self._epistemic_metrics_log: list[dict[str, object]] = []
        self._alerts: list[str] = []

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

    def process_meta_audit(self, audit_reports: Iterable[Mapping[str, float]]) -> dict:
        """Process epistemic audit reports and emit telemetry."""

        reports = list(audit_reports)
        if not reports:
            raise ValueError("audit_reports must contain at least one report")
        metrics = compute_epistemic_metrics(reports)
        self.log_to_ledger("epistemic_metrics", metrics)
        if metrics.get("epistemic_drift", 0.0) > 0.05:
            self.alert("Epistemic drift exceeds threshold")
        return metrics

    def log_to_ledger(self, channel: str, payload: Mapping[str, object]) -> None:
        """Record telemetry payloads for observability or testing."""

        entry = {"channel": channel, "payload": dict(payload)}
        ledger = self._epistemic_ledger
        if ledger is not None and hasattr(ledger, "append_meta"):
            ledger.append_meta(channel, entry["payload"])
        else:
            self._epistemic_metrics_log.append(entry)

    def alert(self, message: str) -> None:
        """Store governance alerts triggered by meta-audits."""

        self._alerts.append(message)

    @property
    def epistemic_metrics_log(self) -> list[dict[str, object]]:
        return list(self._epistemic_metrics_log)

    @property
    def alerts(self) -> list[str]:
        return list(self._alerts)

    @staticmethod
    def _attach_signature(decision: GovernanceDecision, signature: DecisionSignature) -> None:
        decision.timestamp_token = signature.timestamp_token
        decision.signature = signature.signature
