"""
Detects contradictions between representation and efficiency in governance systems.
"""

import json, time, hashlib, random

SAMPLE_DATA = [
    {"region": "City A", "decision_time_days": 12, "voter_turnout": 0.62},
    {"region": "City B", "decision_time_days": 3, "voter_turnout": 0.31},
    {"region": "City C", "decision_time_days": 7, "voter_turnout": 0.55},
]

def sha(x): return hashlib.sha256(x.encode()).hexdigest()

def contradiction_score(turnout, decision_time):
    return round(min(1.0, (1 - turnout) + (decision_time / 10) * 0.2), 2)

def detect_governance_conflicts(data=SAMPLE_DATA):
    out=[]
    for d in data:
        score = contradiction_score(d["voter_turnout"], d["decision_time_days"])
        if score>0.5:
            out.append({
                "region": d["region"],
                "voter_turnout": d["voter_turnout"],
                "decision_time_days": d["decision_time_days"],
                "contradiction_score": score,
                "timestamp": int(time.time()),
                "source_hash": sha(json.dumps(d))
            })
    return out

if __name__=="__main__":
    print(json.dumps(detect_governance_conflicts(), indent=2))