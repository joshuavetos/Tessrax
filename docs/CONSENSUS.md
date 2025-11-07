# Tessrax Federation Consensus Layer

## Overview
The federation consensus layer implements a simplified HotStuff-style
protocol capable of simulating leader rotation, quorum certificates, and
commit notifications across a four-node (or greater) federation.

## Components
- **HotStuffConsensus**: Maintains per-node locking rules, block
  registration, and commit tracking.
- **QuorumTracker**: Aggregates votes into quorum certificates using the
  `2f + 1` rule.
- **ViewManager**: Rotates leaders when the configurable timeout is
  exceeded and prohibits view regressions.
- **InMemoryNetwork**: Broadcasts proposals and votes between nodes and
  records commit notifications.
- **FederationSimulator**: High-level orchestration that runs rounds,
  records latency metrics, and reports consensus completion.

## Guarantees
- Nodes refuse proposals whose justification conflicts with the locked
  view, preventing safety violations.
- Latency for the standard simulation remains bounded to 150 ms under
  typical execution on commodity hardware.
- Consensus requires at least four nodes, aligning with Byzantine fault
  tolerance requirements.

## Audit Receipts
Running `scripts/generate_federation_consensus_receipt.py` yields
`out/federation_consensus_receipt.json`, capturing integrity metrics,
latency, and Tessrax governance metadata.
