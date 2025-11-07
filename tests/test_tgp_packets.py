import math

from tessrax.tgp import (
    CreditTx,
    FederationHeader,
    GovernancePacket,
    decode_cbor,
    decode_json,
    encode_cbor,
    encode_json,
)


def test_deterministic_packet_serialization():
    header = FederationHeader(node_id="node-1", quorum_epoch=5, prev_commit_hash="abc123")
    tx = CreditTx(sender="alice", receiver="bob", amount=42.5)
    receipt = {"status": "ok", "auditor": "kernel"}
    packet = GovernancePacket(
        federation_header=header,
        credit_tx=tx,
        receipt=receipt,
        merkle_inclusion_proof="proof",
        nonce="0001",
    )

    payload = packet.to_dict()
    json_encoded = encode_json(payload)
    cbor_encoded = encode_cbor(payload)

    assert decode_json(json_encoded) == payload
    assert decode_cbor(cbor_encoded) == payload
    assert packet.payload_hash() == packet.payload_hash()
    assert math.isfinite(float(len(json_encoded)))
    assert len(json_encoded) > 0
