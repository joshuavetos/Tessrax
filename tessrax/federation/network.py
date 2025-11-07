"""In-memory network transport used by the consensus simulator."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .consensus import CommitEvent


class InMemoryNetwork:
    """Reliable broadcast network delivering messages instantly."""

    def __init__(self, latency_ms: int = 50) -> None:
        self.latency_ms = latency_ms
        self.nodes: Dict[str, "Node"] = {}
        self.commits: Dict[str, List[CommitEvent]] = defaultdict(list)

    def register(self, node: "Node") -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Node {node.node_id} already registered")
        self.nodes[node.node_id] = node
        node.attach_network(self)

    def broadcast_proposal(self, sender: str, message: "ProposalMessage") -> None:
        for node_id, node in self.nodes.items():
            if node_id == sender:
                continue
            node.receive_proposal(message)

    def broadcast_vote(self, sender: str, message: "VoteMessage") -> None:
        for node in self.nodes.values():
            node.receive_vote(message)

    def notify_commit(self, receiver: str, event: CommitEvent) -> None:
        self.commits[receiver].append(event)


from .node import Node, ProposalMessage, VoteMessage  # noqa: E402 circular

__all__ = ["InMemoryNetwork"]
