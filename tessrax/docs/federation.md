# Tessrax: Federation Protocol

**Version:** 1.0  
**Status:** Draft Specification  
**Last Updated:** 2025-10-05  

---

## 1. Purpose

The **Federation Protocol** defines how multiple Tessrax nodes synchronize receipts, scars, and governance state across distributed environments.  
Its purpose is to guarantee **consistency without centralization**—every participant can independently verify truth while remaining autonomous.

---

## 2. Design Goals

| Goal | Description |
|------|--------------|
| **Decentralized Verification** | No single node is authoritative; consensus emerges from signed proofs. |
| **Eventual Consistency** | All peers converge toward the same ledger state through periodic reconciliation. |
| **Tamper Evidence** | Any modification to past state is detectable through hash mismatch. |
| **Resilience** | Network partition or node failure does not compromise truth continuity. |
| **Auditability** | Every synchronization event leaves a verifiable receipt. |

---

## 3. Core Concepts

### 3.1 Federation Node
A Tessrax instance capable of signing, verifying, and broadcasting receipts.  
Each node maintains:
- Its **local ledger** (append-only SQLite or equivalent)
- A **public signing key** for validation
- A **peer registry** for communication

### 3.2 Synchronization Event
A signed transaction that encapsulates a delta between two ledgers:
```json
{
  "type": "SYNC_EVENT",
  "origin": "node_A",
  "target": "node_B",
  "timestamp": "2025-10-05T18:20:00Z",
  "delta_merkle_root": "<sha256>",
  "entries": ["<receipt_hash_1>", "<receipt_hash_2>", "..."],
  "signature": "<origin_signature>"
}