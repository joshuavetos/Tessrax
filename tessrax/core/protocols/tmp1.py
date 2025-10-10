
```python
"""
Tessrax Minimal Protocol (TMP-1) Reference Implementation
Implements the universal contradiction-governance loop.
"""

import hashlib, json
S, L = set(), []

def conflict(a,b):
    """Domain-agnostic contradiction rule."""
    return ("not "+a)==b or ("not "+b)==a

def step(claim):
    """Execute one governance cycle on a single claim."""
    S.add(claim)
    C = {(a,b) for a in S for b in S if a!=b and conflict(a,b)}
    σ = 1 - len(C)/max(1,len(S))
    route = "accept" if σ>0.8 else "reconcile" if σ>=0.5 else "reset"
    if route=="reconcile":
        S.difference_update({x for p in C for x in p})
    if route=="reset":
        S.clear()
    prev = L[-1]["hash"] if L else ""
    rec = {
        "claim": claim,
        "contradictions": list(C),
        "stability": σ,
        "route": route,
        "state": list(S),
        "prev_hash": prev,
        "hash": hashlib.sha256((json.dumps(S,sort_keys=True)+prev).encode()).hexdigest()
    }
    L.append(rec)
    return rec

if __name__ == "__main__":
    for c in ["system stable", "not system stable", "performance high"]:
        print(step(c))