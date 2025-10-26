"""Statistical mechanics helpers for Tessrax physics models."""

from __future__ import annotations

import numpy as np


class StatisticalMechanics:
    """Provide combinatorial and entropy utilities."""

    def __init__(self, k_B: float = 1.0) -> None:
        self.k_B = k_B

    def microstate_count(self, states: int, branching: int = 3) -> int:
        """Return the count of accessible microstates."""

        return branching**states

    def boltzmann_entropy(self, omega: float) -> float:
        """Return the Boltzmann entropy for the supplied microstate count."""

        return float(self.k_B * np.log(omega))
