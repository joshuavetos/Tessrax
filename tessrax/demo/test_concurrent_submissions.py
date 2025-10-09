"""
Concurrency Test — ensures ledger integrity under parallel writes.
"""
import json
import concurrent.futures
from pathlib import Path
from ce_mod_66 import detect_contradictions, score_stability

def write_claim(i, ledger_path: Path):
    claim = [{"agent": f"A{i}", "claim": f"Claim-{i}", "type": "epistemic"}]
    G = detect_contradictions(claim)
    stability = score_stability(G)
    entry = {"id": i, "stability": stability}
    with open(ledger_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

def test_concurrent_claim_submissions(tmp_path):
    ledger_file = tmp_path / "ledger.jsonl"
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(write_claim, i, ledger_file) for i in range(50)]
        for f in futures: f.result()
    
    # Check file integrity
    with open(ledger_file) as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 50
    ids = {r["id"] for r in lines}
    assert len(ids) == 50  # all unique, no race overwrite