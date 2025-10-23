# Governance Protocols

Tessrax metabolises contradictions according to four cooperating protocols. Each protocol exposes measurable controls so operators can calibrate transparency, trust, and responsiveness.

## Memory Protocol

- **Purpose** – Preserve an immutable account of system reasoning and data lineage.
- **Core Rules**
  - Append-only receipts signed by module keys.
  - Snapshot checkpoints every N contradictions with Merkle proofs.
  - Retain raw inputs alongside derived claims for reproducibility.
- **Key Metrics** – Ledger retention horizon, receipt verification latency, snapshot integrity ratio.

## Metabolism Protocol

- **Purpose** – Convert contradictions into clarified knowledge without exhausting the system.
- **Core Rules**
  - Compute Clarity Fuel from resolved contradictions: `fuel = 12 × detachment_score^1.5`.
  - Route contradictions through severity queues (informational, operational, existential).
  - Escalate to Governance when aggregate tension exceeds configured limits.
- **Key Metrics** – Average contradiction half-life, queue backlog, Clarity Fuel balance.

## Governance Protocol

- **Purpose** – Coordinate decision making with quorum rules and policy enforcement.
- **Core Rules**
  - Weighted quorum requiring ≥ 80% agreement for irreversible actions.
  - Policy bundles define required evidence, decision deadlines, and recovery actions.
  - Conflict arbitration chooses between remediation, acceptance, or reset workflows.
- **Key Metrics** – Quorum attainment rate, policy breach frequency, reset frequency.

## Trust Protocol

- **Purpose** – Maintain transparency and accountability for internal and external observers.
- **Core Rules**
  - Compute trust scores per agent with exponential decay and reinforcement on verified contributions.
  - Publish public receipts with redactable payloads via zero-knowledge commitments.
  - Trigger audit beacons when trust falls below guardrails or ledger anomalies occur.
- **Key Metrics** – Trust score volatility, disclosure coverage, audit closure time.

## Protocol Interplay

| Trigger | Memory Response | Metabolism Response | Governance Response | Trust Response |
| --- | --- | --- | --- | --- |
| New contradiction detected | Log contradiction packet with provenance. | Queue into metabolic pipeline and compute energy. | Evaluate against policy thresholds. | Update observer digest, notify stakeholders. |
| Clarity Fuel deficit | Flag ledger marker for resource scarcity. | Accelerate resolution cadence. | Consider rebalancing policy weights. | Publish transparency note to rebuild trust. |
| Ledger verification failure | Halt ledger appends and seal segment. | Pause metabolic actions pending audit. | Convene emergency quorum to restore integrity. | Issue public incident notification. |

## Escalation Paths

1. **Operational Contradiction** → Metabolism resolves automatically → Memory archives receipt → Trust publishes status summary.
2. **High-Severity Contradiction** → Governance quorum convenes → Memory checkpoints ledger → Trust invites external observers.
3. **Protocol Conflict** (e.g., policy vs. trust) → Governance selects precedence rule → Memory logs rationale → Metabolism recalibrates thresholds.
