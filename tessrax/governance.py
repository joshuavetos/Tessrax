"""Governance kernel simulation."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from types import SimpleNamespace

try:  # pragma: no cover - optional dependency with heavyweight backend
    from sentence_transformers import SentenceTransformer, util
except Exception:  # pragma: no cover - graceful fallback when unavailable
    SentenceTransformer = None  # type: ignore[assignment]
    util = None  # type: ignore[assignment]

from tessrax.governance_security import DecisionSignature, SignatureAuthority
from tessrax.meta_integrity.analytics import compute_epistemic_metrics
from tessrax.types import ContradictionRecord, GovernanceDecision

logger = logging.getLogger(__name__)

_MODEL: SentenceTransformer | None = None


def _load_model() -> SentenceTransformer | None:
    """Lazily load the sentence transformer used for contradiction typing."""

    global _MODEL
    if _MODEL is not None or SentenceTransformer is None:
        return _MODEL
    try:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:  # pragma: no cover - defensive path when model load fails
        logger.warning(
            "Falling back to heuristic contradiction classification: %s", exc
        )
        _MODEL = None
    return _MODEL


def _token_overlap_ratio(text_a: str, text_b: str) -> float:
    tokens_a = {token.lower() for token in text_a.split() if token}
    tokens_b = {token.lower() for token in text_b.split() if token}
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return overlap / union if union else 0.0


def _claim_to_text(claim: object) -> str:
    if (
        hasattr(claim, "subject")
        and hasattr(claim, "metric")
        and hasattr(claim, "value")
    ):
        subject = getattr(claim, "subject", "")
        metric = getattr(claim, "metric", "")
        value = getattr(claim, "value", "")
        return f"{subject} {metric} {value}".strip()
    return str(claim)


def classify_contradiction(text_a: str, text_b: str) -> str:
    """Lightweight semantic contradiction classifier."""

    model = _load_model()
    if model is not None and util is not None:
        try:
            emb_a, emb_b = model.encode([text_a, text_b], convert_to_tensor=True)
            score = util.pytorch_cos_sim(emb_a, emb_b).item()
        except Exception as exc:  # pragma: no cover - catch runtime backend errors
            logger.warning(
                "Semantic model error – reverting to heuristic scoring: %s", exc
            )
            score = None
        else:
            if score < 0.4:
                return "semantic"
            if score < 0.75:
                return "procedural"
            return "normative"
    ratio = _token_overlap_ratio(text_a, text_b)
    if ratio < 0.2:
        return "semantic"
    if ratio < 0.6:
        return "procedural"
    return "normative"


def route_to_governance_lane(contradiction_record: ContradictionRecord) -> str:
    """Route contradiction to the proper governance lane.

    The routing strategy prioritises semantic or logical contradictions with
    sufficient confidence, otherwise delegating to review or general queues.
    """

    ctype = getattr(contradiction_record, "contradiction_type", None)
    claims: Sequence = getattr(contradiction_record, "claims", ())
    if (ctype is None or not isinstance(ctype, str)) and len(claims) == 2:
        claim_texts = (_claim_to_text(claims[0]), _claim_to_text(claims[1]))
        ctype = classify_contradiction(*claim_texts)

    confidence = getattr(contradiction_record, "confidence", 0.5) or 0.5
    if ctype in ("semantic", "logical") and confidence >= 0.7:
        lane = "high_priority_lane"
    elif ctype in ("procedural", "normative"):
        lane = "review_lane"
    else:
        lane = "general_lane"

    logger.debug(
        "Routing contradiction type=%s confidence=%s → %s",
        ctype or "unknown",
        round(confidence, 3),
        lane,
    )
    return lane


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
        return 12 * (detachment_score**1.5)

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
    def _attach_signature(
        decision: GovernanceDecision, signature: DecisionSignature
    ) -> None:
        decision.timestamp_token = signature.timestamp_token
        decision.signature = signature.signature

    def ingest_record(self, record: Mapping[str, object] | ContradictionRecord) -> str:
        """Route inbound records to the appropriate governance lane."""

        if isinstance(record, Mapping):
            raw_meta = (
                dict(record.get("meta", {}))
                if isinstance(record.get("meta"), Mapping)
                else {}
            )
            event_type = record.get("event_type")
            lane_source = SimpleNamespace(
                **{k: v for k, v in record.items() if k.isidentifier()}
            )
        else:
            meta_attr = getattr(record, "meta", {})
            raw_meta = dict(meta_attr) if isinstance(meta_attr, Mapping) else {}
            event_type = getattr(record, "event_type", None)
            lane_source = record

        if raw_meta.get("synthetic"):
            lane = "sandbox_lane"
        elif event_type == "FEDERATED_EXCHANGE":
            lane = "federated_import_lane"
        else:
            lane = route_to_governance_lane(lane_source)

        ledger = self._epistemic_ledger
        if ledger is not None and hasattr(ledger, "append_meta"):
            ledger.append_meta(
                "governance_ingest",
                {
                    "event_type": "GOVERNANCE_INGEST",
                    "record_id": getattr(record, "id", None)
                    or (record.get("id") if isinstance(record, Mapping) else None),
                    "lane": lane,
                    "meta": raw_meta,
                },
            )

        return lane
