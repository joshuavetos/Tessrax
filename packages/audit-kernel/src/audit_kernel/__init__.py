"""Stub audit-kernel package for deterministic dependency locking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = ["AuditKernel", "AuditInsight", "__version__"]

__version__ = "0.1.0"


@dataclass(frozen=True)
class AuditInsight:
    statement: str
    confidence: float


class AuditKernel:
    """Minimal stand-in for the production audit kernel."""

    def assess(self, record: object) -> AuditInsight:  # pragma: no cover - trivial stub
        return AuditInsight(statement=f"Stub insight for {record}", confidence=0.5)

    def batch(self, records: Sequence[object]) -> Sequence[AuditInsight]:  # pragma: no cover - trivial stub
        return [self.assess(record) for record in records]
