"""Coverage-oriented tests for the governance kernel integration layer."""
from __future__ import annotations

from typing import Any

import pytest

from tessrax.core.governance_kernel import GovernanceKernel


class _StubKernel(GovernanceKernel):
    """Subclass to expose protected summarisation helper for direct testing."""

    def public_summary(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return self._summarise_field_evidence(records)


@pytest.fixture
def sample_records() -> list[dict[str, Any]]:
    return [
        {
            "id": "rec-1",
            "category": "climate",
            "summary": "Contradiction detected in field notes",
            "key_findings": ["contradiction", "alignment drop"],
            "alignment": {"policy_reference": "P-1", "score": 0.4},
        },
        {
            "id": "rec-2",
            "category": "climate",
            "summary": "Routine observation",
            "key_findings": ["stability"],
            "alignment": {"policy_reference": "P-1", "score": 0.6},
        },
    ]


def test_field_evidence_summary_flags_contradictions(sample_records: list[dict[str, Any]]) -> None:
    kernel = _StubKernel(auto_integrate=False)
    summary = kernel.public_summary(sample_records)

    assert summary["total_records"] == 2
    assert summary["category_counts"]["climate"] == 2
    assert summary["alignment_scores"]["P-1"] == pytest.approx(0.5)
    assert summary["contradiction_cases"] == ["rec-1"]


def test_integrate_field_evidence_refreshes_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    kernel = GovernanceKernel(auto_integrate=False)
    calls: list[list[dict[str, Any]]] = []

    def _load() -> list[dict[str, Any]]:
        payload = [
            {
                "id": "rec-1",
                "category": "resilience",
                "summary": "No contradictions",
                "key_findings": ["healthy"],
                "alignment": {"policy_reference": "P-9", "score": 0.9},
            }
        ]
        calls.append(payload)
        return payload

    monkeypatch.setattr("tessrax.core.governance_kernel.load_field_evidence", _load)

    first = kernel.integrate_field_evidence(refresh=True)
    assert first["category_counts"] == {"resilience": 1}

    # Invoke again without refresh to ensure cached data is reused.
    second = kernel.integrate_field_evidence()
    assert second == first
    assert len(calls) == 1
