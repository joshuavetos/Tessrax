"""
End-to-End Integration Test — Tessrax-Core
Simulates a full run: agent claims → contradiction detection → routing → ledger logging.
"""
import json
from ce_mod_66 import detect_contradictions, score_stability
from governance_kernel import route

def test_full_contradiction_flow(tmp_path):
    # 1. Generate synthetic agent data
    agent_claims = [
        {"agent": "GPT", "claim": "Option A", "type": "normative"},
        {"agent": "Gemini", "claim": "Option B", "type": "normative"},
        {"agent": "Copilot", "claim": "Option A", "type": "normative"},
    ]
    
    # 2. Detect contradictions + compute stability
    G = detect_contradictions(agent_claims)
    result = route(G)
    
    # 3. Persist to temporary ledger
    log_entry = {
        "stability_index": result["stability"],
        "governance_lane": result["lane"]
    }
    ledger_file = tmp_path / "ledger.jsonl"
    with open(ledger_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # 4. Verify ledger integrity
    with open(ledger_file) as f:
        lines = f.readlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert 0.0 <= record["stability_index"] <= 1.0
    assert record["governance_lane"] in ["autonomic","deliberative","constitutional","behavioral_audit"]