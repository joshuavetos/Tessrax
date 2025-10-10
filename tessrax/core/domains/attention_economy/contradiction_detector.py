"""
attention_contradiction_detector.py
Tessrax v0.1 — Detects contradictions in the attention economy.
Core tension: engagement maximization vs. user well-being.
"""

import json, time, hashlib, random

SAMPLE_DATA = [
    {"platform": "TikTok", "avg_session_min": 92, "wellbeing_score": 0.34},
    {"platform": "YouTube", "avg_session_min": 63, "wellbeing_score": 0.46},
    {"platform": "Insta", "avg_session_min": 58, "wellbeing_score": 0.41},
    {"platform": "Threads", "avg_session_min": 40, "wellbeing_score": 0.65},
]

def sha256(x: str) -> str:
    return hashlib.sha256(x.encode()).hexdigest()

def contradiction_score(session_min, wellbeing):
    """Higher score → stronger divergence between engagement and well-being."""
    return round(min(1.0, (session_min / 60) * (1 - wellbeing)), 2)

def detect_attention_conflicts(data=SAMPLE_DATA):
    receipts = []
    for d in data:
        score = contradiction_score(d["avg_session_min"], d["wellbeing_score"])
        if score > 0.5:
            receipts.append({
                "platform": d["platform"],
                "session_min": d["avg_session_min"],
                "wellbeing_score": d["wellbeing_score"],
                "contradiction_score": score,
                "timestamp": int(time.time()),
                "source_hash": sha256(json.dumps(d, sort_keys=True))
            })
    return receipts

if __name__ == "__main__":
    print(json.dumps(detect_attention_conflicts(), indent=2))