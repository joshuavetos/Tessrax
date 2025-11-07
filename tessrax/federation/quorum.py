"""Quorum utilities for the Tessrax federation consensus layer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple


@dataclass(frozen=True)
class QuorumCertificate:
    """Represents a HotStuff quorum certificate."""

    view: int
    block_hash: str
    signers: frozenset[str]

    def has_quorum(self, total_nodes: int) -> bool:
        required = (2 * total_nodes) // 3 + 1
        return len(self.signers) >= required


class QuorumTracker:
    """Aggregates votes into quorum certificates."""

    def __init__(self, total_nodes: int) -> None:
        if total_nodes < 1:
            raise ValueError("total_nodes must be positive")
        self.total_nodes = total_nodes
        self._votes: Dict[Tuple[int, str], Set[str]] = {}

    def record_vote(self, view: int, block_hash: str, voter: str) -> QuorumCertificate | None:
        key = (view, block_hash)
        bucket = self._votes.setdefault(key, set())
        bucket.add(voter)
        certificate = QuorumCertificate(view=view, block_hash=block_hash, signers=frozenset(bucket))
        if certificate.has_quorum(self.total_nodes):
            return certificate
        return None

    def reset_view(self, view: int) -> None:
        keys_to_remove = [key for key in self._votes if key[0] == view]
        for key in keys_to_remove:
            del self._votes[key]


__all__ = ["QuorumCertificate", "QuorumTracker"]
