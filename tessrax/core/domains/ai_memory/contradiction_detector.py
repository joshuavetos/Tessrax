"""
memory_contradiction_detector.py
Tessrax v0.1 — Detects contradictions in AI memory updates.
Core tension: coherence vs. contradiction retention.
"""

import json, time, hashlib, random

# --- Sample synthetic data (old vs. new memory states) -------------------
SAMPLE_MEMORY_LOG = [
    {"key": "ethics_policy", "old": "safe deployment requires delays", "new": "accelerate deployment to compete"},
    {"key": "alignment_doc", "old": "hallucination rate 0.5%", "new": "hallucination rate 3%"},
    {"key": "training_data", "old": "source verified", "new": "source unverified"},
]

def sha256(x: str) -> str:
    return hashlib.sha256(x.encode()).hexdigest()

def detect_memory_conflicts(entries=SAMPLE_MEMORY_LOG):
    """Return receipts for all overwritten contradictions."""
    receipts = []
    for e in entries:
        if e["old"] != e["new"]:
            score = round(random.uniform(0.6, 0.95), 2)
            receipts.append({
                "memory_key": e["key"],
                "old_value": e["old"],
                "new_value": e["new"],
                "contradiction_score": score,
                "timestamp": int(time.time()),
                "source_hash": sha256(json.dumps(e, sort_keys=True))
            })
    return receipts

if __name__ == "__main__":
    print(json.dumps(detect_memory_conflicts(), indent=2))