"""Consensus primitives implementing a simplified HotStuff protocol."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Optional

from .quorum import QuorumCertificate

GENESIS_HASH = "genesis"


def _hash_dict(payload: Dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class Block:
    """Block proposal exchanged between federation nodes."""

    view: int
    parent_hash: str
    payload: Dict[str, object]
    proposer: str

    def block_hash(self) -> str:
        payload = {
            "view": self.view,
            "parent_hash": self.parent_hash,
            "payload": self.payload,
            "proposer": self.proposer,
        }
        return _hash_dict(payload)


@dataclass(frozen=True)
class Vote:
    """Vote emitted by a node for a block."""

    voter: str
    view: int
    block_hash: str


@dataclass
class CommitEvent:
    """Represents a committed block and related metadata."""

    block_hash: str
    view: int
    proposer: str


class HotStuffConsensus:
    """Per-node consensus state machine."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.locked_view = 0
        self.locked_block_hash = GENESIS_HASH
        self._known_blocks: Dict[str, Block] = {}
        self.committed: list[str] = [GENESIS_HASH]

    def register_block(self, block: Block) -> None:
        self._known_blocks[block.block_hash()] = block

    def validate_proposal(self, block: Block, justify: QuorumCertificate) -> bool:
        if block.parent_hash != justify.block_hash:
            return False
        if justify.view < self.locked_view:
            return False
        if self.locked_block_hash not in {block.parent_hash, GENESIS_HASH} and block.parent_hash != self.locked_block_hash:
            return False
        return True

    def vote_for_block(self, block: Block, justify: QuorumCertificate) -> Optional[Vote]:
        if not self.validate_proposal(block, justify):
            return None
        self.locked_view = max(self.locked_view, block.view)
        self.locked_block_hash = block.block_hash()
        return Vote(voter=self.node_id, view=block.view, block_hash=block.block_hash())

    def try_commit(self, certificate: QuorumCertificate) -> Optional[CommitEvent]:
        block = self._known_blocks.get(certificate.block_hash)
        if block is None:
            return None
        block_hash = certificate.block_hash
        if block_hash not in self.committed:
            self.committed.append(block_hash)
            return CommitEvent(block_hash=block_hash, view=block.view, proposer=block.proposer)
        return None

    def latest_commit(self) -> str:
        return self.committed[-1]


__all__ = [
    "GENESIS_HASH",
    "Block",
    "Vote",
    "CommitEvent",
    "HotStuffConsensus",
]
