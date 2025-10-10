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
4. Reference Python Implementation
import hashlib, json
S, L = set(), []

def conflict(a,b):
    return ("not "+a)==b or ("not "+b)==a

def step(claim):
    S.add(claim)
    C = {(a,b) for a in S for b in S if a!=b and conflict(a,b)}
    σ = 1 - len(C)/max(1,len(S))
    route = "accept" if σ>0.8 else "reconcile" if σ>=0.5 else "reset"
    if route=="reconcile": S.difference_update({x for p in C for x in p})
    if route=="reset": S.clear()
    prev = L[-1]["hash"] if L else ""
    rec = {"claim":claim,"contradictions":list(C),"stability":σ,"route":route,
           "state":list(S),"prev_hash":prev,
           "hash":hashlib.sha256((json.dumps(S,sort_keys=True)+prev).encode()).hexdigest()}
    L.append(rec)
    return rec
5. Formal Specification (Symbolic Form)
∀c∈C:
  Δ = { (a,b) | a,b∈S ∧ conflict(a,b) }
  σ = 1 − |Δ| / |S|
  r = f(σ) = accept|reconcile|reset
  S' = apply(r,S,Δ)
  L ← L ⧺ { S', Δ, σ, r, hash(L[-1]) }
6. Behavioral Guarantees
   •   Self-stabilizing: Repeated cycles converge to a contradiction-minimal state.
   •   Auditable: Every mutation of state is hash-chained.
   •   Extensible: Any domain may redefine conflict() and summary().
   •   Language-neutral: Translatable into JSON, Python, or symbolic form.

⸻

7. Quick-Start Demo
python - <<'PY'
from Core.protocols.tmp1 import step
for c in ["system stable", "not system stable", "performance high"]:
    print(step(c))
PY
8. Citation

Tessrax LLC. (2025). Tessrax Minimal Protocol (TMP-1): Universal contradiction-governance loop.
Version 1.0. https://github.com/joshuavetos/Tessrax