"""
Aggregate TMP-1 audits to find global patterns
"""
import json
from collections import Counter
from glob import glob

def analyze_audits(audit_files):
    all_contradictions, stability_scores = [], []

    for file in audit_files:
        audit = json.load(open(file))
        all_contradictions.extend(audit["contradictions_found"])
        stability_scores.append(audit["stability_score"])

    contradiction_types = Counter(c["type"] for c in all_contradictions)
    avg_stability = sum(stability_scores) / len(stability_scores)

    return {
        "total_audits": len(audit_files),
        "total_contradictions": len(all_contradictions),
        "avg_stability": round(avg_stability, 3),
        "contradiction_breakdown": dict(contradiction_types),
        "high_risk_audits": sum(1 for s in stability_scores if s < 0.5)
    }

if __name__ == "__main__":
    results = analyze_audits(glob("Documents/examples/audits/*.json"))
    print(json.dumps(results, indent=2))