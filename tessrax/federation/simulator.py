"""Federation consensus simulator for integration testing."""
from __future__ import annotations

import time
from statistics import mean
from textwrap import dedent
from typing import Dict, Iterable, List

from .network import InMemoryNetwork
from .node import Node
from .view_manager import ViewManager


class FederationSimulator:
    """Runs a deterministic simulation across federation nodes."""

    def __init__(self, node_ids: Iterable[str], view_timeout_ms: int = 150) -> None:
        self.node_ids = list(node_ids)
        if len(self.node_ids) < 4:
            raise ValueError("Simulator expects at least four nodes for Byzantine quorum")
        self.network = InMemoryNetwork()
        self.view_manager = ViewManager(self.node_ids, view_timeout_ms=view_timeout_ms)
        self.nodes: Dict[str, Node] = {}
        for node_id in self.node_ids:
            node = Node(node_id, total_nodes=len(self.node_ids))
            self.network.register(node)
            self.nodes[node_id] = node
        self.latencies: List[float] = []

    def run_round(self, payload: Dict[str, object]) -> None:
        leader = self.view_manager.leader()
        view = self.view_manager.current_view + 1
        start = time.perf_counter()
        self.nodes[leader].propose(payload=payload, view=view)
        self.view_manager.force_advance(view)
        latency_ms = (time.perf_counter() - start) * 1000.0
        self.latencies.append(latency_ms)
        message = dedent(
            """
            Leader {leader} proposed view {view} with latency {latency:.2f} ms.
            Payload keys: {payload_keys}
            """
        ).strip().format(
            leader=leader,
            view=view,
            latency=latency_ms,
            payload_keys=sorted(payload.keys()),
        )
        print(message)

    def average_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return round(mean(self.latencies), 3)

    def consensus_reached(self) -> bool:
        committed_counts = [len(self.network.commits[node_id]) for node_id in self.node_ids]
        return all(count > 0 for count in committed_counts)


__all__ = ["FederationSimulator"]
