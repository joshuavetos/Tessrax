from nacl import signing

from tessrax.tgp import (
    CreditTx,
    FederationHeader,
    GovernanceClient,
    GovernancePacket,
    PacketRouter,
    ReplayProtection,
    Signer,
    SignatureVerifier,
)


def test_client_routes_packet():
    key = signing.SigningKey.generate()
    signer = Signer(key._seed)
    router = PacketRouter(SignatureVerifier(ReplayProtection()))
    received = {}

    def handler(packet: GovernancePacket):
        received["hash"] = packet.payload_hash()
        return packet.credit_tx.amount

    router.register("GovernancePacket", handler)

    header = FederationHeader(node_id="node-1", quorum_epoch=1, prev_commit_hash="genesis")
    tx = CreditTx(sender="alice", receiver="bob", amount=5.0)
    packet = GovernancePacket(
        federation_header=header,
        credit_tx=tx,
        receipt={"status": "ok"},
        merkle_inclusion_proof="proof",
        nonce="nonce-123",
    )

    client = GovernanceClient(signer=signer, router=router)
    result = client.submit(packet)
    assert result == 5.0
    assert "hash" in received
