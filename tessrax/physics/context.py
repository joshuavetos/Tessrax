"""Contextual physics helpers for contradiction energy calculations."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


class ContextualStiffness:
    """Estimate how rigid a subject/metric context is."""

    DOMAIN_STIFFNESS: dict[str, float] = {
        "legal": 0.9,
        "emissions": 0.7,
        "opinion": 0.2,
        "forecast": 0.3,
    }

    def compute(self, subject: str, metric: str, temporal_distance: float) -> float:
        """Return contextual stiffness adjusted by temporal decay."""

        base = self.DOMAIN_STIFFNESS.get(metric, 0.5)
        decay = np.exp(-0.1 * temporal_distance)
        return float(base * decay)


class AgentAlignment:
    """Track the historical alignment between information sources."""

    def __init__(self) -> None:
        self._history: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])

    @staticmethod
    def _ordered_pair(agent_a: str, agent_b: str) -> tuple[str, str]:
        """Return a stable ordering for agent identifiers."""

        return (agent_a, agent_b) if agent_a <= agent_b else (agent_b, agent_a)

    def update(self, agent_a: str, agent_b: str, agreed: bool) -> None:
        """Record whether two agents agreed on a comparison."""

        key = self._ordered_pair(agent_a, agent_b)
        record = self._history[key]
        record[1] += 1
        if agreed:
            record[0] += 1

    def compute(self, agent_a: str, agent_b: str) -> float:
        """Return the historical alignment ratio for two agents."""

        key = self._ordered_pair(agent_a, agent_b)
        record = self._history.get(key)
        if record is None:
            return 0.5
        successes, trials = record
        if trials == 0:
            return 0.5
        return successes / trials
