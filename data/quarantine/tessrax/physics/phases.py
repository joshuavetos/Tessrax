"""Governance phase logic driven by contextual integrity and drift."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GovernancePhase(Enum):
    """Discrete phase labels for governance dynamics."""

    STABLE = "stable"
    CAUTIOUS = "cautious"
    CRITICAL = "critical"


@dataclass(slots=True)
class PhaseTransition:
    """Compute the governance phase from integrity and drift inputs."""

    INTEGRITY_CRITICAL: float = 0.7
    INTEGRITY_CAUTIOUS: float = 0.9
    DRIFT_CRITICAL: float = 0.1
    DRIFT_CAUTIOUS: float = 0.05

    def compute_phase(self, integrity: float, drift: float) -> GovernancePhase:
        """Return the governance phase for the provided signals."""

        if integrity < self.INTEGRITY_CRITICAL or drift > self.DRIFT_CRITICAL:
            return GovernancePhase.CRITICAL
        if integrity < self.INTEGRITY_CAUTIOUS or drift > self.DRIFT_CAUTIOUS:
            return GovernancePhase.CAUTIOUS
        return GovernancePhase.STABLE
