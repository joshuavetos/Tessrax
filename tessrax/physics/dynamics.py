"""Lightweight dynamical models for reconciliation feedback loops."""

from __future__ import annotations


class SystemDynamics:
    """Evaluate simple energy and trust evolution equations."""

    def __init__(
        self, gamma: float = 0.1, alpha: float = 0.5, beta: float = 0.3
    ) -> None:
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def energy_evolution(self, energy: float, _time: float, forcing) -> float:
        """Compute energy dissipation with an external forcing function."""

        return -self.gamma * energy + float(forcing(_time))

    def trust_evolution(
        self,
        state: tuple[float, ...],
        _time: float,
        reliability: float,
        energy: float,
        drift: float,
    ) -> tuple[float]:
        """Return the instantaneous change in trust/integrity."""

        d_integrity_dt = self.alpha * (reliability - energy) - self.beta * drift
        return (d_integrity_dt,)
