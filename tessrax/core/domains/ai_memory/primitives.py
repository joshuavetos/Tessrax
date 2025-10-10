"""
memory_primitives.py
Tessrax v0.1 — Governance primitives for epistemic metabolism.
"""

def conflict_density(total_keys: int, conflicting_keys: int) -> float:
    """Proportion of memory entries with contradiction flags."""
    return round(conflicting_keys / max(total_keys, 1), 3)

def coherence_penalty(overwrites: int, retained: int) -> float:
    """Information loss per overwrite relative to retained knowledge."""
    return round(overwrites / max(retained + overwrites, 1), 3)

def provenance_retention_yield(retained: int, provenance_links: int) -> float:
    """Value of keeping provenance-rich contradictions."""
    yield_pct = min(1.0, (provenance_links / max(retained, 1)) * 0.8)
    return round(yield_pct, 3)

if __name__ == "__main__":
    print("Conflict density:", conflict_density(100, 23))
    print("Coherence penalty:", coherence_penalty(40, 60))
    print("Provenance yield:", provenance_retention_yield(60, 45))