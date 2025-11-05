"""Federated network simulation utilities."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable

from tessrax.core.contracts.license import embed_license
from tessrax.core.contracts.token import TokenBank
from tessrax.core.ledger import append_entry

from .node import LEDGER_PATH as DEFAULT_LEDGER_PATH
from .node import Node
from .quorum import merge_events


async def _initialise_nodes(n: int) -> list[Node]:
    nodes = [Node(f"node_{index}") for index in range(n)]
    for node in nodes:
        await node.link_peers(nodes)
    return nodes


async def simulate_cluster(
    n: int = 3,
    rounds: int = 3,
    ledger_path: Path | str | None = None,
) -> bool:
    """Simulate a small cluster and write quorum events to the ledger."""

    if n <= 0:
        raise ValueError("Cluster size must be positive")
    if rounds <= 0:
        raise ValueError("Number of rounds must be positive")

    ledger_target = Path(ledger_path) if ledger_path else DEFAULT_LEDGER_PATH
    ledger_target.parent.mkdir(parents=True, exist_ok=True)

    nodes = await _initialise_nodes(n)

    bank = TokenBank()

    for round_index in range(rounds):
        events = await asyncio.gather(*(node.run_round() for node in nodes))
        quorum = merge_events(events)
        quorum["round"] = round_index + 1
        append_entry(quorum, ledger_target)
        if ledger_target != DEFAULT_LEDGER_PATH:
            append_entry(quorum, DEFAULT_LEDGER_PATH)
        reward = bank.mint("cluster", 5, "federated_quorum")
        embed_license(reward)
        await asyncio.sleep(0.05)

    return True


__all__ = ["simulate_cluster"]
