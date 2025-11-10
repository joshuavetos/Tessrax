#!/usr/bin/env python3
"""Generate receipt for Tessrax Governance Protocol core validation."""
from __future__ import annotations

import json
import platform
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tessrax.tgp import (
    CreditTx,
    FederationHeader,
    GovernancePacket,
    PacketRouter,
    ReplayProtection,
    Signer,
    SignatureVerifier,
)
from nacl import signing


def run_checks() -> float:
    key = signing.SigningKey.generate()
    signer = Signer(key._seed)
    router = PacketRouter(SignatureVerifier(ReplayProtection()))
    header = FederationHeader(node_id="node-1", quorum_epoch=10, prev_commit_hash="genesis")
    tx = CreditTx(sender="alice", receiver="bob", amount=50.0)
    packet = GovernancePacket(
        federation_header=header,
        credit_tx=tx,
        receipt={"status": "ok"},
        merkle_inclusion_proof="proof",
        nonce="receipt",
    )

    handled = {}

    def handler(inner_packet: GovernancePacket) -> float:
        handled["hash"] = inner_packet.payload_hash()
        return inner_packet.credit_tx.amount

    router.register("GovernancePacket", handler)
    payload = packet.to_dict()
    envelope = signer.sign(payload, nonce=packet.nonce, issued_at=time.time())
    router.route(packet, envelope)
    checks = ["hash" in handled, handled.get("hash") == packet.payload_hash()]
    return round(sum(checks) / len(checks), 2)


def main() -> None:
    integrity_score = run_checks()
    if integrity_score < 0.95:
        raise SystemExit("TESS MODE BLOCK: Integrity below protocol threshold")
    receipt = {
        "timestamp": time.time(),
        "runtime_info": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "integrity_score": integrity_score,
        "status": "pass",
        "auditor": "Tessrax Governance Kernel v16",
        "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
    }
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    receipt["signature"] = signing.SigningKey.generate().sign(payload).signature.hex()
    output_path = Path("out/tgp_core_receipt.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("DLK-VERIFIED receipt stored at", output_path)


if __name__ == "__main__":
    main()
