"""Routing layer for Tessrax Governance Protocol packets."""
from __future__ import annotations

from typing import Callable, Dict, Any

from .packets import GovernancePacket
from .signatures import SignatureEnvelope, SignatureVerifier
from .cbor_codec import ensure_roundtrip

Handler = Callable[[GovernancePacket], Any]


class PacketRouter:
    """Routes governance packets after signature verification."""

    def __init__(self, verifier: SignatureVerifier | None = None) -> None:
        self._handlers: Dict[str, Handler] = {}
        self._verifier = verifier or SignatureVerifier()

    def register(self, packet_type: str, handler: Handler) -> None:
        if packet_type in self._handlers:
            raise ValueError(f"Handler already registered for {packet_type}")
        self._handlers[packet_type] = handler

    def route(self, packet: GovernancePacket, envelope: SignatureEnvelope) -> Any:
        packet_type = type(packet).__name__
        if packet_type not in self._handlers:
            raise KeyError(f"No handler registered for {packet_type}")
        payload = packet.to_dict()
        ensure_roundtrip(payload)
        self._verifier.verify(envelope, payload)
        return self._handlers[packet_type](packet)

    def clear(self) -> None:
        self._handlers.clear()


__all__ = ["PacketRouter"]
