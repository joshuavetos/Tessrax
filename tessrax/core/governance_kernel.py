"""Governance kernel extensions with field evidence integration."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from tessrax.data.evidence_loader import load_field_evidence
from tessrax.governance import GovernanceKernel as BaseGovernanceKernel


class GovernanceKernel(BaseGovernanceKernel):
    """Governance kernel that bootstraps field evidence archives."""

    def __init__(self, *args: Any, auto_integrate: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._field_evidence: list[dict[str, Any]] = []
        self._field_evidence_summary: dict[str, Any] = {}
        if auto_integrate:
            self.integrate_field_evidence()

    @property
    def field_evidence(self) -> list[dict[str, Any]]:
        """Return the cached field evidence entries."""

        return [entry.copy() for entry in self._field_evidence]

    def integrate_field_evidence(self, *, refresh: bool = False) -> dict[str, Any]:
        """Load field evidence and compute contradiction/policy alignment indicators."""

        if not self._field_evidence or refresh:
            self._field_evidence = load_field_evidence()
            self._field_evidence_summary = self._summarise_field_evidence(
                self._field_evidence
            )
        return dict(self._field_evidence_summary)

    def _summarise_field_evidence(
        self, records: Iterable[Mapping[str, Any]]
    ) -> dict[str, Any]:
        category_counts: dict[str, int] = defaultdict(int)
        alignment_scores: dict[str, list[float]] = defaultdict(list)
        contradiction_cases: list[str] = []

        for record in records:
            category = str(record.get("category", "uncategorised"))
            category_counts[category] += 1

            alignment = record.get("alignment")
            if isinstance(alignment, Mapping):
                policy = str(alignment.get("policy_reference", "unspecified"))
                score = alignment.get("score")
                if isinstance(score, (int, float)):
                    alignment_scores[policy].append(float(score))

            searchable_text = " ".join(
                [
                    str(record.get("summary", "")),
                    " ".join(record.get("key_findings", [])),
                ]
            ).lower()
            if "contradiction" in searchable_text:
                record_id = str(record.get("id", "")) or category
                contradiction_cases.append(record_id)

        averaged_alignment = {
            policy: sum(scores) / len(scores)
            for policy, scores in alignment_scores.items()
            if scores
        }

        return {
            "total_records": sum(category_counts.values()),
            "category_counts": dict(category_counts),
            "alignment_scores": averaged_alignment,
            "contradiction_cases": contradiction_cases,
        }


__all__ = ["GovernanceKernel"]
