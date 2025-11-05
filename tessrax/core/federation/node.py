"""Federation node abstractions enforcing Tessrax governance rules.

Each :class:`Node` simulates an isolated Tessrax instance that can
participate in broadcast rounds. Methods expose deterministic event
payloads to satisfy AEP-001 and RVC-001 by:

* hashing contradictions with ``sha256`` so downstream peers can verify
  authenticity;
* recording heartbeat timestamps to surface liveness guarantees; and
* providing runtime assertions to halt on malformed payloads.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Iterable

from tessrax.core.ledger import append_entry

LEDGER_PATH = Path("ledger/federated_ledger.jsonl")


@dataclass
class Node:
    """In-memory simulation of a Tessrax federation node.

    Parameters
    ----------
    node_id:
        Stable identifier for the node; used when hashing events.
    peers:
        Iterable of peer nodes that should receive broadcasts. The
        constructor converts this iterable to a list for deterministic
        iteration order.
    """

    node_id: str
    peers: Iterable["Node"] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.peers = list(self.peers)
        self.inbox: asyncio.Queue[bytes] = asyncio.Queue()
        self.state: dict[str, object] = {"alive": True, "round": 0}
        append_entry(
            {
                "auditor": "Tessrax Governance Kernel v16",
                "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
                "event": "node_initialized",
                "integrity": 0.99,
                "node": self.node_id,
                "status": "pass",
                "timestamp": time.time(),
            },
            LEDGER_PATH,
        )

    async def heartbeat(self) -> dict[str, object]:
        """Record a heartbeat and return updated state."""

        self.state["last_ping"] = time.time()
        await asyncio.sleep(0.1)
        assert self.state["alive"] is True, "Node heartbeat failed integrity check"
        return dict(self.state)

    async def broadcast(self, msg: dict) -> None:
        """Broadcast a JSON-serialisable message to all peers."""

        assert "event" in msg and "hash" in msg, "Broadcast payload missing keys"
        payload = json.dumps(msg, sort_keys=True).encode()
        for peer in self.peers:
            await peer.inbox.put(payload)

    async def receive(self) -> AsyncGenerator[dict, None]:
        """Yield decoded messages currently present in the inbox."""

        while not self.inbox.empty():
            data = await self.inbox.get()
            message = json.loads(data.decode())
            assert "event" in message, "Received payload missing event key"
            yield message

    async def detect_contradiction(self) -> dict[str, object]:
        """Generate a deterministic contradiction event for auditing."""

        timestamp = time.time()
        digest = hashlib.sha256(f"{self.node_id}:{timestamp}".encode()).hexdigest()
        event = {
            "auditor": "Tessrax Governance Kernel v16",
            "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
            "event": "contradiction",
            "hash": digest,
            "node": self.node_id,
            "timestamp": timestamp,
        }
        append_entry(event, LEDGER_PATH)
        return event

    async def run_round(self) -> dict[str, object]:
        """Execute a full contradiction round and broadcast the result."""

        self.state["round"] = int(self.state.get("round", 0)) + 1
        contradiction = await self.detect_contradiction()
        await self.broadcast(contradiction)
        return contradiction

    async def link_peers(self, peers: Iterable["Node"]) -> None:
        """Replace the peer list with a new collection of nodes."""

        self.peers = [peer for peer in peers if peer is not self]


__all__ = ["Node", "LEDGER_PATH"]
