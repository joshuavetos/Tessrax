"""Energy modelling utilities for Tessrax contradictions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ContradictionEnergy:
    """Simple harmonic-style model for contradiction energy."""

    k_B: float = 1.0

    def compute_kappa(
        self,
        contextual_stiffness: float,
        agent_alignment: float,
        base_probability: float,
    ) -> float:
        """Return the effective spring constant for a contradiction."""

        return contextual_stiffness * agent_alignment * base_probability

    def contradiction_energy(self, delta: float, kappa: float) -> float:
        """Compute the potential energy stored in the contradiction gap."""

        return 0.5 * kappa * (delta**2)
