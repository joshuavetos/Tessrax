"""
housing_contradiction_detector.py
Tessrax v0.1 — Housing Contradiction Detection Module
Detects financial vs. physical durability tension in U.S. housing data.
"""

import hashlib, json, random, time

# --- Synthetic sample data for demo ---------------------------------------
# In production, replace with real CSV import (Zillow, FRED, insurance, etc.)

SAMPLE_DATA = [
    {"location": "Travis County, TX", "avg_home_age": 32, "avg_refi_per_decade": 3.8, "avg_material_lifespan": 60},
    {"location": "Cook County, IL", "avg_home_age": 47, "avg_refi_per_decade": 2.4, "avg_material_lifespan": 80},
    {"location": "Maricopa County, AZ", "avg_home_age": 24, "avg_refi_per_decade": 4.2, "avg_material_lifespan": 45},
    {"location": "Orange County, CA", "avg_home_age": 38, "avg_refi_per_decade": 3.0, "avg_material_lifespan": 70},
]

# --- Helpers ---------------------------------------------------------------
def sha256(x: str) -> str:
    return hashlib.sha256(x.encode()).hexdigest()

def contradiction_score(velocity: float, durability: float) -> float:
    """Higher score means stronger contradiction between churn and longevity."""
    return round(min(1.0, (velocity / (durability / 10)) / 2.0), 2)

# --- Core Detector ---------------------------------------------------------
def detect_contradictions(data=SAMPLE_DATA):
    receipts = []
    for d in data:
        v = d["avg_refi_per_decade"]
        dur = d["avg_material_lifespan"]
        score = contradiction_score(v, dur)
        if score > 0.5:
            payload = {
                "location": d["location"],
                "velocity": v,
                "durability_index": round(dur / 100, 2),
                "contradiction_score": score,
                "timestamp": int(time.time()),
                "source_hash": sha256(json.dumps(d, sort_keys=True)),
            }
            receipts.append(payload)
    return receipts

if __name__ == "__main__":
    contradictions = detect_contradictions()
    print(json.dumps(contradictions, indent=2))
