"""
Bayesian Trust Calibration Model
Implements adaptive trust decay and redemption cycles.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass


@dataclass
class BayesianTrust:
    """Maintain a Bayesian trust score with decay and redemption mechanics."""

    alpha: float = 1.0
    beta: float = 1.0
    decay_rate: float = 0.98

    def __post_init__(self) -> None:
        self._last_update = datetime.datetime.now(datetime.timezone.utc)

    def update(self, success: bool) -> None:
        """Update the posterior parameters applying exponential decay."""

        self.alpha *= self.decay_rate
        self.beta *= self.decay_rate
        if success:
            self.alpha += 1
        else:
            self.beta += 1
        self._last_update = datetime.datetime.now(datetime.timezone.utc)

    @property
    def last_update(self) -> datetime.datetime:
        return self._last_update

    @property
    def score(self) -> float:
        total = self.alpha + self.beta
        return round(self.alpha / total, 3) if total else 0.0

    def redeem(self, factor: float = 1.1) -> None:
        """Boost trust to reward positive remediation behaviour."""

        self.alpha *= factor
        self._last_update = datetime.datetime.now(datetime.timezone.utc)

    def to_dict(self) -> dict[str, object]:
        """Serialise the trust state for logging or dashboards."""

        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "trust_score": self.score,
            "timestamp": self._last_update.isoformat(),
        }
