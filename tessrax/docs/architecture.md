# Tessrax Architecture

_This document outlines the architecture of Tessrax, describing its primary components (“primitives”), their interactions, and the governance pipeline that orchestrates decision-making and auditability._

---

## 1. Architectural Primitives

### 1.1 Receipts
- **Purpose:** Tamper-evident, cryptographically signed proofs of computation.
- **Structure:** Contains hashes of code, inputs, outputs, timestamp, executor ID, and signature.
- **Core Functions:**
  - `generate_receipt(code, inputs, outputs, executor_id)`
  - `verify_receipt(receipt, code, inputs, outputs)`
- **Role:** Anchors each computation in a verifiable, auditable record.

### 1.2 Memory
- **Purpose:** Key/value store with provenance and contradiction detection.
- **Features:**
  - Supports provenance tracking for each entry.
  - Detects contradictions (multiple values for the same key).
- **Role:** Maintains evolving state and highlights inconsistencies.

### 1.3 CSV (Contrastive Scar Verification)
- **Purpose:** Verification primitive for stress-testing reproducibility.
- **Functions:**
  - `candidate_output(x)`: Deterministic output.
  - `contrast_output(x)`: Slightly perturbed output.
  - `verify(x)`: Checks if candidate matches contrast.
- **Role:** Ensures verification logic is robust against minor output drift.

### 1.4 Ledger
- **Purpose:** Append-only, tamper-evident log of events.
- **Features:**
  - Each event appended is cryptographically linked (Merkle root).
  - Supports integrity verification.
- **Role:** Core audit log for governance and traceability.

### 1.5 Agents
- **Purpose:** Autonomous or human-supervised decision makers.
- **Features:**
  - Agents propose actions via `Agent.act()`.
  - Human-in-the-loop can approve or reject decisions.
- **Role:** Blends automation with human oversight.

### 1.6 Sandbox
- **Purpose:** Secure, isolated code execution.
- **Features:**
  - Enforces CPU and memory limits.
  - Captures all outputs (stdout/stderr).
  - Returns structured results (success, output, error, timing).
- **Role:** Prevents runaway or malicious code, provides auditable results.

### 1.7 Quorum & Dissent
- **Quorum:** Multi-signature threshold for collective decision enforcement.
- **Dissent:** Explicitly records disagreement or failed quorum.

### 1.8 Federation
- **Purpose:** Multi-node simulation and coordination.
- **Features:**
  - Broadcasts events, synchronizes state, manages cache.
- **Role:** Ensures distributed consistency and resilience.

---

## 2. Governance Pipeline (Demo Flow)

The Tessrax demo flow connects these primitives into a governance pipeline as follows:

1. **Decision Proposal:**  
   An agent proposes an action.
2. **Human Approval:**  
   Human-in-the-loop approves or rejects. Dissent is recorded if rejected.
3. **Sandboxed Execution:**  
   Approved actions execute in a secure sandbox.
4. **Receipt Generation:**  
   Execution results are wrapped in a verifiable receipt.
5. **Ledger Append:**  
   The receipt is appended to the ledger, updating the Merkle root.
6. **Quorum Validation:**  
   Collective agreement is checked. Failures are logged as dissent.
7. **Federation Propagation:**  
   Receipts and state changes are broadcast to peer nodes.

---

## 3. Data Flow Diagram

```
+-------+        +--------------+        +---------+
| Agent | -----> | Human Review | -----> | Sandbox |
+-------+        +--------------+        +---------+
     |                   |                     |
     |                   v                     v
     |             [Approve/Reject]       [Execution]
     |                   |                     |
     v                   v                     |
+-----------+       +-----------+              |
|  Dissent  |       |  Receipt  | <------------
+-----------+       +-----------+              |
                          |                    |
                          v                    v
                      +---------+        +-----------------+
                      | Ledger  | -----> | Quorum/Dissent  |
                      +---------+        +-----------------+
                          |
                          v
                    +-------------+
                    | Federation  |
                    +-------------+
```

---

## 4. Key Design Principles

- **Transparency:** All computations are auditable and tamper-evident.
- **Contradiction Handling:** State inconsistencies are detected and logged.
- **Human-in-the-Loop:** Blends automation with required human oversight.
- **Extensibility:** Architecture supports the addition of new primitives.
- **Distributed Resilience:** Federation enables multi-node deployments.

---

## 5. Limitations & Future Work

- **Current system is for demonstration; not hardened for production or multi-user concurrency.**
- **Planned enhancements:**
  - Stronger consistency and access control.
  - Improved sandbox isolation.
  - Blockchain-based log anchoring.
  - Production-grade federation.

---

_Last updated: 2025-10-05_
