"""
Failure Mode Tests — corrupted ledger, invalid JSON, bad config.
"""
import json
from verify_ledger import verify_chain

def test_corrupted_ledger(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    good = {"timestamp": "t1", "hash": "aaa", "prev_hash": "zzz"}
    bad = "{malformed json line"
    
    with open(ledger, "w") as f:
        f.write(json.dumps(good) + "\n")
        f.write(bad + "\n")
    
    # verify_chain should fail gracefully
    try:
        ok, _ = verify_chain(str(ledger))
    except Exception:
        ok = False
    assert not ok

def test_missing_config(monkeypatch):
    import config_loader
    monkeypatch.setattr(config_loader, "CONFIG_PATH", "nonexistent.json")
    try:
        config_loader.load_config()
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised