"""
Tessrax Constellation Engine v1.0
---------------------------------
Detects and registers capability constellations when the required
mythics are fully formed.  Integrates with the ledger/receipt system.

Dependencies:
    - receipts.create_receipt
    - ledger.append_entry
"""

import json, time, hashlib
from pathlib import Path
from typing import Dict, List, Any
from .receipts import create_receipt
from .ledger import append_entry

# ---------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------
class Constellation:
    """Static definition of a constellation blueprint."""
    def __init__(self, name: str, capability: str, mythics_required: List[str]):
        self.name = name
        self.capability = capability
        self.mythics_required = mythics_required

    def to_dict(self):
        return {
            "name": self.name,
            "capability": self.capability,
            "mythics_required": self.mythics_required
        }


class ConstellationUnlock:
    """Runtime event produced when a constellation forms."""
    def __init__(self, constellation: Constellation, mythics_present: List[str]):
        self.constellation = constellation
        self.mythics_present = mythics_present
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.hash = self._hash_self()

    def _hash_self(self):
        raw = json.dumps({
            "constellation": self.constellation.to_dict(),
            "mythics_present": sorted(self.mythics_present),
            "timestamp": self.timestamp
        }, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def to_event(self) -> Dict[str, Any]:
        return {
            "event": "CONSTELLATION_UNLOCK",
            "timestamp": self.timestamp,
            "capability": self.constellation.capability,
            "constellation_name": self.constellation.name,
            "mythics_present": self.mythics_present,
            "hash": self.hash
        }

# ---------------------------------------------------------------------
# Detection + registration
# ---------------------------------------------------------------------
def detect_constellation(user_mythics: List[str],
                         constellation_db: List[Constellation]) -> List[ConstellationUnlock]:
    """
    Check which constellations can form given a user's formed mythics.

    Args:
        user_mythics: list of mythic names (strings) currently formed
        constellation_db: list of Constellation objects

    Returns:
        List of ConstellationUnlock objects for matches
    """
    formed = set(m.lower() for m in user_mythics)
    unlocks = []
    for c in constellation_db:
        required = {m.lower() for m in c.mythics_required}
        if required.issubset(formed):
            unlocks.append(ConstellationUnlock(c, list(required)))
    return unlocks


def register_unlock(unlock: ConstellationUnlock,
                    private_key_hex: str,
                    executor_id: str = "tessrax.constellation") -> Dict[str, Any]:
    """
    Generate a signed receipt for a constellation unlock and append it to the ledger.
    """
    payload = unlock.to_event()
    receipt = create_receipt(private_key_hex, payload, executor_id)
    append_entry({
        "type": "constellation_unlock",
        "payload": payload,
        "receipt": receipt
    })
    return receipt

# ---------------------------------------------------------------------
# Demo (safe to run)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from .receipts import generate_keypair
    from .constellations_data import CONSTELLATION_DB

    priv, pub = generate_keypair()
    user_mythics = [
        "Heavier Than Air",
        "Control vs Stability",
        "Power vs Weight",
        "Theory vs Practice",
        "Progress vs Survival"
    ]

    unlocks = detect_constellation(user_mythics, CONSTELLATION_DB)
    for u in unlocks:
        r = register_unlock(u, priv)
        print(json.dumps(r, indent=2))