"""Data structures for Tessrax Governance Protocol packets."""
from __future__ import annotations

from dataclasses import dataclass, field
import uuid
from typing import Any, Dict

from .cbor_codec import hash_payload


def _canonicalize_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively sort mappings to achieve deterministic serialization."""
    canonical: Dict[str, Any] = {}
    for key in sorted(mapping):
        value = mapping[key]
        if isinstance(value, dict):
            canonical[key] = _canonicalize_mapping(value)
        elif isinstance(value, list):
            canonical[key] = [
                _canonicalize_mapping(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            canonical[key] = value
    return canonical


@dataclass(frozen=True)
class FederationHeader:
    """Envelope metadata describing the federation state."""

    node_id: str
    quorum_epoch: int
    prev_commit_hash: str

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "node_id": self.node_id,
            "quorum_epoch": int(self.quorum_epoch),
            "prev_commit_hash": self.prev_commit_hash,
        }
        return _canonicalize_mapping(payload)


@dataclass(frozen=True)
class CreditTx:
    """Represents a federated credit transfer transaction."""

    sender: str
    receiver: str
    amount: float

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": float(self.amount),
        }
        return _canonicalize_mapping(payload)


@dataclass(frozen=True)
class GovernancePacket:
    """Combines headers, transactions, and cryptographic receipts."""

    federation_header: FederationHeader
    credit_tx: CreditTx
    receipt: Dict[str, Any]
    merkle_inclusion_proof: str
    nonce: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "federation_header": self.federation_header.to_dict(),
            "credit_tx": self.credit_tx.to_dict(),
            "receipt": _canonicalize_mapping(dict(self.receipt)),
            "merkle_inclusion_proof": self.merkle_inclusion_proof,
            "nonce": self.nonce,
        }
        return _canonicalize_mapping(payload)

    def payload_hash(self) -> str:
        """Return a deterministic hash of the packet payload."""
        return hash_payload(self.to_dict())


__all__ = ["FederationHeader", "CreditTx", "GovernancePacket"]
