import pytest
from nacl import signing

from tessrax.tgp import ReplayProtection, Signer, SignatureVerifier, GovernancePacket, FederationHeader, CreditTx


@pytest.fixture()
def sample_packet() -> GovernancePacket:
    header = FederationHeader(node_id="node-1", quorum_epoch=7, prev_commit_hash="deadbeef")
    tx = CreditTx(sender="alice", receiver="bob", amount=10.0)
    receipt = {"status": "ok"}
    return GovernancePacket(
        federation_header=header,
        credit_tx=tx,
        receipt=receipt,
        merkle_inclusion_proof="merkle",
        nonce="abc",
    )


def test_signature_verification(sample_packet: GovernancePacket):
    key = signing.SigningKey.generate()
    signer = Signer(key._seed)
    verifier = SignatureVerifier(ReplayProtection())

    payload = sample_packet.to_dict()
    envelope = signer.sign(payload, nonce=sample_packet.nonce, issued_at=1.0)
    assert verifier.verify(envelope, payload)

    payload["receipt"]["status"] = "tampered"
    with pytest.raises(ValueError):
        verifier.verify(envelope, payload)


def test_replay_protection(sample_packet: GovernancePacket):
    key = signing.SigningKey.generate()
    signer = Signer(key._seed)
    replay = ReplayProtection()
    verifier = SignatureVerifier(replay)

    payload = sample_packet.to_dict()
    envelope = signer.sign(payload, nonce="nonce-1", issued_at=1.0)
    assert verifier.verify(envelope, payload)
    with pytest.raises(ValueError):
        verifier.verify(envelope, payload)
