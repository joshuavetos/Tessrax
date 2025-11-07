# Tessrax Governance Protocol (TGP) v1.0

## Overview
The Tessrax Governance Protocol provides deterministic packet
serialization, Ed25519 signature envelopes, and replay protection for
federated governance actions. The specification captured here reflects
the v1.0 implementation integrated into the Tessrax repository.

## Serialization
- JSON serialization uses canonical ordering via sorted keys.
- CBOR serialization uses canonical encoding to guarantee byte-level
  determinism across platforms.
- All packets MUST round-trip between JSON and CBOR without mutation.

## Packet Types
### `FederationHeader`
Captures the proposer identity, federation epoch, and previously
committed hash.

### `CreditTx`
Represents a credit transfer describing the sender, receiver, and amount
in federation units.

### `GovernancePacket`
Combines the header, credit transaction, governance receipt, and a
Merkle inclusion proof. Every packet exposes a deterministic `payload_hash`
used for signing and routing.

## Signatures
- Ed25519 signatures are produced over the tuple
  `(payload_hash, nonce, issued_at)`.
- Replay protection is enforced through nonce tracking scoped by public
  key.
- Signature verification MUST fail if the payload hash mismatches or the
  nonce has been observed before.

## Routing & Client
- `PacketRouter` validates signatures and dispatches packets to
  registered handlers.
- `GovernanceClient` signs packets with a local keypair and submits
  them to the router.

## Audit Receipts
Execution of `scripts/generate_tgp_receipt.py` produces an auditable
receipt at `out/tgp_core_receipt.json` containing integrity metrics and a
protocol-compliant signature block.
