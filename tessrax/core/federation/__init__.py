"""Federated governance primitives for Tessrax.

This module exposes the core interfaces required to coordinate
multi-node consensus rounds without external dependencies. All
implementations satisfy Tessrax governance clauses by providing
runtime validation hooks and deterministic hashing for quorum events.
"""

from __future__ import annotations

from .network import simulate_cluster
from .node import Node
from .quorum import merge_events

__all__ = ["Node", "merge_events", "simulate_cluster"]
