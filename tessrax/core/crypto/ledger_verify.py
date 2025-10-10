"""
Automatically verifies all signed receipts in the ledger.
"""

import json
from core.crypto.signature import verify_receipt
from core.crypto.key_vault import load_keys

LEDGER_PATH = "ledger.jsonl"

def audit_ledger():
    keys = load_keys()
    vk = keys["public_key"]
    valid, total = 0, 0
    with open(LEDGER_PATH) as f:
        for line in f:
            rec = json.loads(line)
            total += 1
            if verify_receipt(rec, vk): valid += 1
    print(f"Verified {valid}/{total} receipts ({valid/total*100:.1f}% valid)")

if __name__ == "__main__":
    audit_ledger()