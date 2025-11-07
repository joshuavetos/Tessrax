"""Minimal governance client for submitting packets."""
from __future__ import annotations

import time
from typing import Any

from .packets import GovernancePacket
from .signatures import Signer, SignatureEnvelope
from .router import PacketRouter
from .cbor_codec import ensure_roundtrip


class GovernanceClient:
    """Client capable of signing and dispatching governance packets."""

    def __init__(self, signer: Signer, router: PacketRouter) -> None:
        self._signer = signer
        self._router = router

    def _build_envelope(self, packet: GovernancePacket, nonce: str | None = None) -> SignatureEnvelope:
        payload = packet.to_dict()
        ensure_roundtrip(payload)
        nonce = nonce or packet.nonce
        issued_at = time.time()
        return self._signer.sign(payload, nonce=nonce, issued_at=issued_at)

    def submit(self, packet: GovernancePacket, nonce: str | None = None) -> Any:
        envelope = self._build_envelope(packet, nonce=nonce)
        return self._router.route(packet, envelope)


__all__ = ["GovernanceClient"]
