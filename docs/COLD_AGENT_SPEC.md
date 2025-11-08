# Cold Agent (Epistemic Contradiction Resolver) Specification

**Auditor:** Tessrax Governance Kernel v16  
**Clauses:** AEP-001, POST-AUDIT-001, RVC-001, EAC-001

## Overview

The Cold Agent runtime unifies schema validation, contradiction detection,
state canonicalization, receipt emission, ledger integrity, and audit
verification into a deterministic pipeline.  The runtime is fully
contained within `tessrax/cold_agent/` and exposes the following
components:

| Module | Responsibility |
| --- | --- |
| `schema_validator.py` | Deterministic payload validation with canonical hashing |
| `state_canonicalizer.py` | Maintains canonical state, emits pre/post hashes, exposes diffs |
| `contradiction_engine.py` | Detects conflicting updates and emits contradiction records |
| `receipt_emitter.py` | Produces auditable receipts with SHA-256 signatures |
| `integrity_ledger.py` | Append-only ledger with Merkle root computation |
| `audit_capsule.py` | Verifies ledger integrity using Merkle proofs |
| `bench.py` | Reference runner that orchestrates the full pipeline |

The runtime outputs `out/cold_agent_run_receipt.json`, which contains a
self-auditing summary of each execution including runtime metrics,
Merkle roots, signatures, and clause identifiers.

## Deterministic Metrics

The reference runner records the following metrics for each event:

| Metric | Description |
| --- | --- |
| `C_e` | Contradiction engine availability (1.0 when the engine runs) |
| `L_p` | Count of contradictions detected for the current event |
| `R_r` | Receipt emitter readiness (1.0 when receipt persisted) |
| `S_c` | State canonicalizer availability (1.0 when update applied) |
| `O_v` | Overall pipeline continuity (1.0 when event processed) |
| `index` | Zero-based index of the processed event |

## Execution Guarantees

1. **Determinism** — All hash computations use sorted JSON serialization
   and SHA-256 to guarantee reproducible receipts.
2. **Runtime Verification** — Every component validates its inputs and
   raises explicit errors when invariants fail, satisfying RVC-001.
3. **Auditability** — Receipts include Tessrax Governance metadata and a
   derived signature.  The ledger exposes a Merkle root consumed by the
   audit capsule.
4. **Receipts-First Compliance** — Each execution writes
   `out/cold_agent_run_receipt.json` with `timestamp`, `runtime_info`,
   `integrity_score`, `status`, `signature`, `auditor`, and `clauses`.

## Validation Flow

1. Events are validated and hashed by `SchemaValidator`.
2. `StateCanonicalizer` applies each event and captures pre/post hashes.
3. `ContradictionEngine` compares the event with the previous state and
   reports contradictions.
4. `ReceiptEmitter` constructs a signed receipt and appends it to the
   `IntegrityLedger`.
5. The ledger computes a Merkle root, which is verified by `AuditCapsule`.
6. `bench.main` aggregates the outcomes into an auditable receipt.

A failing audit terminates execution before the receipt is persisted,
ensuring no non-compliant artifacts are recorded.
