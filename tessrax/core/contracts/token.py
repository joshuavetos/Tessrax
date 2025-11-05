"""Token minting primitives for Tessrax federation rewards."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

LEDGER_PATH = Path("ledger/reward_ledger.jsonl")


class TokenBank:
    """Simple proof-of-contribution ledger."""

    def __init__(self, path: Path = LEDGER_PATH):
        self.path = path
        self.balances: dict[str, int] = {}

    def mint(self, actor: str, amount: int, reason: str):
        self.balances[actor] = self.balances.get(actor, 0) + amount
        timestamp = time.time()
        digest_input = f"{actor}:{amount}:{reason}:{timestamp}".encode()
        entry = {
            "event": "reward_mint",
            "actor": actor,
            "amount": amount,
            "reason": reason,
            "timestamp": timestamp,
            "hash": hashlib.sha256(digest_input).hexdigest(),
        }
        self._append(entry)
        return entry

    def _append(self, entry):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def balance(self, actor: str) -> int:
        return self.balances.get(actor, 0)
