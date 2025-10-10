# Tessrax Minimal Protocol (TMP-1)
**Version:** 1.0  
**Purpose:** Universal contradiction-governance loop — a minimal, language-agnostic instruction set for any self-auditing system.  
**Author:** Tessrax LLC  
**License:** Open reference specification (public domain)

---

## 1. Overview
TMP-1 defines the smallest closed loop required for any system to **observe**, **detect**, **evaluate**, **route**, **act**, and **record** contradictions in its internal state.  
It’s the minimal law of Tessrax — executable, auditable, and domain-agnostic.

---

## 2. Core Cycle

| Phase | Function | Description |
|-------|-----------|-------------|
| **Observe** | `add claim → state` | Ingest new input or statement |
| **Detect** | `contradictions = {(a,b)∈state² | conflict(a,b)}` | Find incompatible pairs |
| **Evaluate** | `stability = 1 - |contradictions| / max(1,|state|)` | Quantify coherence |
| **Route** | thresholds = {0.8, 0.5} | Choose action |
| ↳ >0.8 | → **accept** |
| ↳ 0.5–0.8 | → **reconcile** |
| ↳ <0.5 | → **reset** |
| **Act** | apply route to state |
| **Record** | append event to ledger with hash-chaining |

---

## 3. Canonical JSON Schema

```json
{
  "observe": "add claim → state",
  "detect": "contradictions = {(a,b)∈state² | conflict(a,b)}",
  "evaluate": "stability = 1 - |contradictions| / max(1,|state|)",
  "route": {">0.8": "accept", "0.5–0.8": "reconcile", "<0.5": "reset"},
  "act": {
    "accept": "state ← state ∪ {claim}",
    "reconcile": "state ← state − contradictions + summary(contradictions)",
    "reset": "state ← ∅"
  },
  "record": "ledger.append({state, contradictions, stability, route, hash(prev)})"
}