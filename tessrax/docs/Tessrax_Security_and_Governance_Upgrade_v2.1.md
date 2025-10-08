# Tessrax Security and Governance Upgrade v2.1

## Overview
The Tessrax project has evolved from an advisory prototype into a fortified, demonstrably self-governing system.  
Key upgrades strengthen **cryptographic integrity**, **sandbox isolation**, **active governance enforcement**, and **system resilience**.

---

## 1. Cryptographic Security and Integrity
- **Ed25519 Asymmetric Signatures:** Migrated from HMAC to Ed25519 public-key signatures (PyNaCl).  
  Each receipt is verifiably signed by the private key holder, ensuring non-repudiation.  
- **Hash Chaining & Merkle Anchoring:** All contradictions are chained in an append-only ledger.  
  The Merkle root is anchored externally (e.g., to Git) for tamper evidence.  
- **Merkle Padding Fix:** Replaced node duplication with salted padding to prevent second-preimage collisions.  
- **Revocation Enforcement:** Added `keys/revoked.json` list. Any receipt signed by a revoked key is rejected.  
  Key rotation and revocation are now formally documented.

---

## 2. Sandbox Isolation and Resource Governance
- **RestrictedPython Execution:** Replaced `exec()` with the RestrictedPython library to contain untrusted code.  
- **Memory & Time Limits:** Hard limits (≈100 MB) and execution timeouts prevent DoS and runaway loops.  
- **Defense-in-Depth:** Recommended containerization (Docker, non-privileged user) to mitigate side-channel risks.  
- **Centralized I/O Control:** All file access restricted to `/data/`; path-traversal attempts raise `SandboxViolation`.

---

## 3. Active Governance Enforcement
- **Governance Kernel:** Continuously monitors the ledger and enforces charter rules in real time.  
  Example: validates `PROPOSAL_SUBMITTED` events before emitting `PROPOSAL_VALIDATED`.  
- **Binding Enforcement Hooks:** Git server–side `pre-receive` hook now rejects pushes violating the charter.  
- **Quorum & Dissent Protocols:** Merges require quorum (≥7 reviewers) and ≥2 dissent notes; violations log as `rcpt-0003`.  
- **Failure Observability:** All system failures become auditable `FAILURE_EVENT`s in the ledger via `audit_log.log_failure()`.

---

## 4. Resilience and Observability
- **Atomic Ledger Writes:** Introduced an outbox/aggregator model with file locks for race-free updates.  
- **Prometheus Metrics:** Added counters and histograms (`EVENTS_COUNTER`, `MERKLE_LATENCY`, `FAILURES_COUNTER`) exposed at `/metrics`.  
- **Automated Self-Healing:** Federation nodes detect consensus drift and auto-resync from healthy peers.  
- **Runbooks:** Documented recovery procedures for corruption, key compromise, and federation divergence.

---

### Version Metadata
- **Release:** v2.1  
- **Maintainer:** Joshua Vetos  
- **Date:** 2025-10-08  
- **Signature:** Canonized under SIG-LOCK-001  