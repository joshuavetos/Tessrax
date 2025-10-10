# Tessrax Minimal Protocol (TMP-1)
**Version:** 1.0  
**Purpose:** Universal contradiction-governance loop — a minimal, language-agnostic instruction set for any self-auditing system.  
**Author:** Tessrax LLC  
**License:** Open reference specification (public domain)

---

## 1. Overview

The Tessrax Minimal Protocol defines the smallest closed loop required for any system—human, organizational, or artificial—to **observe**, **detect**, **evaluate**, **route**, **act**, and **record** contradictions in its internal state.  
It provides the operational grammar that underlies all Tessrax implementations.

---

## 2. Core Cycle

| Phase | Function | Description |
|-------|-----------|-------------|
| **Observe** | `add claim → state` | Ingest new input, perception, or statement. |
| **Detect** | `contradictions = {(a,b)∈state² | conflict(a,b)}` | Identify incompatible or inverse pairs. |
| **Evaluate** | `stability = 1 - |contradictions| / max(1,|state|)` | Quantify coherence (0 = chaos, 1 = stability). |
| **Route** | thresholds = { 0.8, 0.5 } | Decide appropriate action: |
|  ↳ > 0.8 | → **accept** |
|  ↳ 0.5 – 0.8 | → **reconcile** |
|  ↳ < 0.5 | → **reset** |
| **Act** | Apply chosen route to state: |
|  accept | `state ← state ∪ {claim}` |
|  reconcile | `state ← state - contradictions + summary(contradictions)` |
|  reset | `state ← ∅` |
| **Record** | `ledger.append({state, contradictions, stability, route, hash(prev)})` | Append immutable event to ledger with hash-chaining. |

---

## 3. Canonical JSON Schema

```json
{
  "observe": "add claim → state",
  "detect": "contradictions = {(a,b)∈state² | conflict(a,b)}",
  "evaluate": "stability = 1 - |contradictions| / max(1,|state|)",
  "route": {
    ">0.8": "accept",
    "0.5–0.8": "reconcile",
    "<0.5": "reset"
  },
  "act": {
    "accept": "state ← state ∪ {claim}",
    "reconcile": "state ← state - contradictions + summary(contradictions)",
    "reset": "state ← ∅"
  },
  "record": "ledger.append({state, contradictions, stability, route, hash(prev)})"
}


⸻

4. Reference Python Implementation

import hashlib, json
S, L = set(), []  # state, ledger

def conflict(a,b):  # domain-specific logic
    return ("not "+a)==b or ("not "+b)==a

def step(claim):
    S.add(claim)                                         # observe
    C = {(a,b) for a in S for b in S if a!=b and conflict(a,b)}  # detect
    σ = 1 - len(C)/max(1,len(S))                         # evaluate
    route = "accept" if σ>0.8 else "reconcile" if σ>=0.5 else "reset"  # route
    if route=="reconcile": S.difference_update({x for p in C for x in p})
    if route=="reset": S.clear()                         # act
    prev = L[-1]["hash"] if L else ""
    rec = {"claim":claim,"contradictions":list(C),"stability":σ,
           "route":route,"state":list(S),
           "hash":hashlib.sha256((json.dumps(S,sort_keys=True)+prev).encode()).hexdigest()}
    L.append(rec)                                        # record
    return rec


⸻

5. Formal Specification (Symbolic Form)

∀c∈C:
  Δ = { (a,b) | a,b∈S ∧ conflict(a,b) }
  σ = 1 - |Δ| / |S|
  r = f(σ) = accept|reconcile|reset
  S' = apply(r,S,Δ)
  L ← L ⧺ { S', Δ, σ, r, hash(L[-1]) }


⸻

6. Behavioral Guarantees
   •   Self-stabilizing: repeated cycles converge to a contradiction-minimal state.
   •   Auditable: every mutation of state is hash-chained.
   •   Extensible: any domain may redefine conflict() and summary().
   •   Language-neutral: translatable into JSON, Python, or mathematical grammar.

⸻

7. Quick-Start Demo

python - <<'PY'
from Tessrax_Minimal_Protocol_TMP1 import step
for c in ["system stable", "not system stable", "performance high"]:
    print(step(c))
PY


⸻

8. Summary

TMP-1 compresses the entire Tessrax framework into six verbs and one invariant:

Observe → Detect → Evaluate → Route → Act → Record

This loop, when run iteratively, constitutes the minimal architecture for self-governing intelligence.

⸻

End of Tessrax Minimal Protocol (TMP-1)

