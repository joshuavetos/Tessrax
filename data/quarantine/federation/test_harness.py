"""Tessrax Federated Test Harness (v18.0).

This module instantiates three lightweight federation nodes, introduces a
Byzantine adversary by corrupting ledger receipts, and confirms that the
Merkle roots of the two honest nodes remain equivalent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from typing import Iterable, Mapping

from tessrax.core.ledger import Ledger
from tessrax.core.metabolism.reconcile import ReconciliationEngine


def _merkle_root(receipts: Iterable[Mapping[str, object]]) -> str:
    concatenated = "".join(str(receipt.get("hash", "")) for receipt in receipts)
    return hashlib.sha256(concatenated.encode("utf-8")).hexdigest()


@dataclass
class Node:
    """A simulation node participating in the federation."""

    name: str
    byzantine: bool = False

    def __post_init__(self) -> None:
        self.path = tempfile.mktemp(prefix=f"{self.name}_ledger_", suffix=".jsonl")
        self.ledger = Ledger(self.path)
        self.engine = ReconciliationEngine(self.ledger)

    async def run_cycle(self, contradictions: list[Mapping[str, object]]) -> list[dict[str, object]]:
        receipts: list[dict[str, object]] = []
        for record in contradictions:
            clarity = self.engine.reconcile([record])
            receipt = clarity.to_receipt()
            if self.byzantine and random.random() > 0.5:
                receipt = dict(receipt)
                receipt["hash"] = hashlib.sha256(os.urandom(32)).hexdigest()
            self.ledger.append(receipt)
            receipts.append(receipt)
        return receipts

    async def merkle_root(self) -> str:
        return _merkle_root(self.ledger.receipts())


async def simulate_federation() -> dict[str, object]:
    """Run a single-cycle federation simulation and return audit metadata."""

    nodes = [
        Node("alpha"),
        Node("beta"),
        Node("gamma", byzantine=True),
    ]
    contradictions = [
        {"subject": "data", "metric": "coherence", "value": 0.9},
        {"subject": "data", "metric": "coherence", "value": 0.1},
    ]
    await asyncio.gather(*(node.run_cycle(contradictions) for node in nodes))
    roots = {node.name: await node.merkle_root() for node in nodes}
    root_values = list(roots.values())
    consensus_root = max(set(root_values), key=root_values.count)
    byzantine_detected = any(root != consensus_root for root in root_values)
    result = {
        "roots": roots,
        "byzantine_detected": byzantine_detected,
        "consensus_root": consensus_root,
        "auditor": "Tessrax Governance Kernel v16",
        "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "DLK-001"],
    }
    print(json.dumps(result, indent=2))
    return result


__all__ = ["Node", "simulate_federation"]
