# Tessrax Engine – Architecture Overview

## Core Concept
Tessrax is a tamper-evident governance engine. It metabolizes contradictions into structured artifacts, logs them immutably, and enforces continuity through hash-chained handoffs.

---

## Components

### 1. Contradiction Metabolizer
- Input: pairs of contradictory statements
- Output: "fuel units" with deterministic IDs and hashes
- Purpose: Turn tension into computable energy for governance

### 2. Governance Layer
- Signs claims with agent ID + timestamp
- Verifies claims against stored signatures
- Provides tamper-evident receipts

### 3. Scar Registry
- Logs contradictions as scars with lineage
- Metadata: severity, status, parent links
- Builds genealogy of unresolved vs. resolved contradictions

### 4. Continuity Handoffs
- Captures snapshots of state
- Hash-chained to prior handoff for immutability
- Functions as blockchain-like continuity ledger

### 5. Ledger
- Append-only JSONL files (`claims.jsonl`, `handoffs.jsonl`)
- Contains receipts for every governance event
- Supports verification of the entire chain

---

## Flow
1. Contradictions → metabolized into fuel units
2. Fuel units → spawn scars
3. Claims → signed & verified
4. State → sealed in a continuity handoff
5. Ledger → enables audit, replay, and verification

---

## Why It Matters
- **Auditability:** Every action produces receipts
- **Integrity:** Cryptographic chaining prevents tampering
- **Governance:** Contradictions become first-class computable assets
- **Continuity:** Reset-proof via handoff chaining
