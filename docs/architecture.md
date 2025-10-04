Tessrax v1.0 Architecture

This document describes the core primitives of Tessrax v1.0, how they interact, and how the demo flow wires them together into a governance pipeline.

⸻

Primitives

Receipts
   •   Purpose: Provide tamper-evident proof of computation.
   •   Structure: Dict with hashes of code, inputs, outputs, timestamp, executor ID, and signature.
   •   Functions:
      •   generate_receipt(code, inputs, outputs, executor_id)
      •   verify_receipt(receipt, code, inputs, outputs)
   •   Role: Anchor computational results into a verifiable record.

Memory
   •   Purpose: Store key/value pairs with provenance and contradiction detection.
   •   Features:
      •   Adding the same value from different sources does not trigger contradiction.
      •   Adding a different value for the same key sets contradiction=True.
   •   Role: Maintain state across agents and highlight inconsistencies.

CSV (Contrastive Scar Verification)
   •   Purpose: Provide candidate vs. contrast outputs for testing.
   •   Functions:
      •   candidate_output(x) → deterministic 2x.
      •   contrast_output(x) → 2x ± 1 (randomized).
      •   verify(x) → True if candidate == contrast.
   •   Role: Stress primitive verification and reproducibility.

Ledger
   •   Purpose: Append-only log of events with Merkle root for integrity.
   •   Functions:
      •   add_event(event) → append to ledger.
      •   merkle_root() → compute root hash.
   •   Role: Governance backbone, ensuring continuity and tamper evidence.

Agents
   •   Purpose: Autonomous decision-makers.
   •   Agent Base: Agent.act() returns a dict with agent name and decision.
   •   Human in Loop: Wraps decisions with human approval or rejection.
   •   Role: Blend automation with human oversight.

Sandbox
   •   Purpose: Execute code safely with resource limits.
   •   Features:
      •   CPU timeout (signal).
      •   Memory cap (resource).
      •   Captures stdout/stderr.
      •   Returns structured result {success, output, error, time}.
   •   Role: Contain execution and provide auditable results.

Quorum & Dissent
   •   Quorum: Threshold multi-signatures enforce collective decision rules.
   •   Dissent: Recorded when humans or quorum checks reject a proposal.

Federation
   •   Purpose: Multi-node simulation with broadcast, quorum, and cache scrub.
   •   Role: Ensure consistency across distributed nodes.

⸻

Demo Flow

The demo (demo_flow.py) wires primitives into a governance pipeline:
	1.	Agent proposes a decision.
	2.	Human in Loop approves or rejects.
      •   If rejected, dissent is recorded.
	3.	Sandbox executes approved decision.
	4.	Receipts wrap execution result.
	5.	Ledger appends receipt and updates Merkle root.
	6.	Quorum checks collective agreement.
      •   If quorum fails, dissent is recorded.
	7.	Federation propagates receipts and cache scrubs across nodes.

⸻

ASCII Data Flow Diagram

+-------+        +--------------+        +---------+
| Agent | -----> | Human in Loop| -----> | Sandbox |
+-------+        +--------------+        +---------+
     |                   |                     |
     |                   v                     v
     |             [Approve/Reject]       [Execution Result]
     |                   |                     |
     v                   v                     |
+-----------+       +-----------+              |
|  Dissent  |       |  Receipt  | <------------
+-----------+       +-----------+              |
                          |                    |
                          v                    v
                      +---------+        +-------------+
                      | Ledger  | -----> | Quorum/Dissent|
                      +---------+        +-------------+
                          |
                          v
                    +-------------+
                    | Federation  |
                    +-------------+


⸻

Summary
   •   Receipts: Verifiable computation records.
   •   Memory: Contradiction-aware state.
   •   CSV: Verification stress tests.
   •   Ledger: Tamper-evident governance log.
   •   Agents: Decision-makers with human oversight.
   •   Sandbox: Safe execution environment.
   •   Quorum/Dissent: Enforce governance rules.
   •   Federation: Distributed resilience.

Together, these primitives form the Tessrax v1.0 Contradiction Metabolism Engine, demonstrated in demo_flow.py.