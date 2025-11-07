"""Federation node abstraction used by the simulator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .consensus import Block, HotStuffConsensus, Vote, GENESIS_HASH
from .quorum import QuorumCertificate, QuorumTracker


@dataclass
class ProposalMessage:
    block: Block
    justify: QuorumCertificate


@dataclass
class VoteMessage:
    vote: Vote


class Node:
    """Represents a federation node participating in consensus."""

    def __init__(self, node_id: str, total_nodes: int) -> None:
        self.node_id = node_id
        self.consensus = HotStuffConsensus(node_id)
        self.quorum_tracker = QuorumTracker(total_nodes)
        self._network: Optional["InMemoryNetwork"] = None
        self.latest_certificate: Optional[QuorumCertificate] = None
        self.blocks: Dict[str, Block] = {}
        self._genesis_certificate = QuorumCertificate(view=0, block_hash=GENESIS_HASH, signers=frozenset({node_id}))

    def attach_network(self, network: "InMemoryNetwork") -> None:
        self._network = network

    def current_certificate(self) -> QuorumCertificate:
        return self.latest_certificate or self._genesis_certificate

    def propose(self, payload: Dict[str, object], view: int) -> Block:
        justify = self.current_certificate()
        block = Block(view=view, parent_hash=justify.block_hash, payload=payload, proposer=self.node_id)
        self.blocks[block.block_hash()] = block
        self.consensus.register_block(block)
        if self._network is None:
            raise RuntimeError("Node is not attached to a network")
        vote = self.consensus.vote_for_block(block, justify)
        if vote:
            self._network.broadcast_vote(self.node_id, VoteMessage(vote=vote))
        self._network.broadcast_proposal(self.node_id, ProposalMessage(block=block, justify=justify))
        return block

    def receive_proposal(self, message: ProposalMessage) -> None:
        block, justify = message.block, message.justify
        self.blocks[block.block_hash()] = block
        self.consensus.register_block(block)
        vote = self.consensus.vote_for_block(block, justify)
        if vote and self._network is not None:
            self._network.broadcast_vote(self.node_id, VoteMessage(vote=vote))

    def receive_vote(self, message: VoteMessage) -> None:
        vote = message.vote
        qc = self.quorum_tracker.record_vote(vote.view, vote.block_hash, vote.voter)
        if qc:
            self.latest_certificate = qc
            commit = self.consensus.try_commit(qc)
            if commit and self._network is not None:
                self._network.notify_commit(self.node_id, commit)


from .network import InMemoryNetwork  # noqa: E402 circular

__all__ = ["Node", "ProposalMessage", "VoteMessage"]
