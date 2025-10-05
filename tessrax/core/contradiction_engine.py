# tessrax/core/contradiction_engine.py
"""
Tessrax Contradiction Engine v4.7 (Hardened / Corrected)
----------------------------------------------------
Secure contradiction-processing runtime:
✓ Verified receipts (secure-by-default)
✓ Sandboxed, durable quarantine with fail-safe handling
✓ Signed contradiction records (delegates chaining to ledger)
✓ Optional scar metabolism
✓ Rule registration and robust error handling
"""

import os
import json
import time
import hashlib
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Callable, List, Optional

from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder
from prometheus_client import Counter

from tessrax.core.receipts import verify_receipt, NonceRegistry, RevocationRegistry
from tessrax.core.resource_guard import ResourceMonitor, ensure_in_sandbox

# (Metrics, Exceptions, Helpers, and Dataclasses remain the same as the previous audited version)

class ContradictionEngine:
    """
    Hardened Contradiction Engine with corrected audit findings.
    """
    def __init__(
        self,
        *,
        ledger,  # SQLiteLedger instance required
        ruleset: Optional[List[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]]] = None,
        signing_key_hex: str,
        nonce_registry: NonceRegistry,
        revocation_registry: RevocationRegistry,
        name: str = "contradiction_engine",
        verify_strict: bool = True,
        quarantine_path: str = "data/quarantine.jsonl",
        metabolize_fn: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    ):
        if not all([ledger, signing_key_hex, nonce_registry, revocation_registry]):
            raise ContradictionEngineError(
                "Ledger, signing_key, NonceRegistry, and RevocationRegistry are required."
            )

        self.name = name
        self.ruleset = ruleset or []
        self.ledger = ledger
        self.nonce_registry = nonce_registry
        self.revocation_registry = revocation_registry
        self.verify_strict = verify_strict
        self.metabolize_fn = metabolize_fn
        self._lock = threading.Lock()

        self.signing_key = SigningKey(signing_key_hex, encoder=HexEncoder)
        self.verify_key = self.signing_key.verify_key.encode(encoder=HexEncoder).decode()

        sandbox_root = Path("data/sandbox")
        self._quarantine_path = ensure_in_sandbox(Path(quarantine_path), sandbox_root)

    def _emit(self, contradiction: Dict[str, Any]) -> None:
        """Sign and emit contradiction, delegating chaining to the ledger."""
        base = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": "contradiction",
            "payload": contradiction,
        }
        # CORRECTED: Remove prev_hash logic. Ledger maintains its own chain.
        signed = self._sign_payload(base)
        self.ledger.add_event(signed)

        if self.metabolize_fn:
            try:
                self.metabolize_fn(contradiction)
            except Exception as e:
                print(f"[WARN] Metabolism function failed: {e}")

        CONTRADICTION_EVENTS_PROCESSED.inc()

    def run_batch(self, events: List[Dict[str, Any]]) -> None:
        """Processes a batch of events, halting on critical quarantine failures."""
        with self._lock:
            for ev in events:
                try:
                    self._run_once_unlocked(ev)
                except QuarantineViolation as e:
                    print(f"[CRITICAL] Halting engine due to quarantine failure: {e}")
                    self.stop()
                    break

    def verify_contradiction_chain(self) -> bool:
        """
        CORRECTED: Delegates full-chain verification to the ledger.
        """
        if not hasattr(self.ledger, "verify_chain"):
            raise ContradictionEngineError(
                "Ledger missing required `verify_chain()` method."
            )
        return self.ledger.verify_chain()

    def get_stats(self) -> Dict[str, Any]:
        """Returns basic operational statistics."""
        all_events = self.ledger.get_all_events(verify=False)
        contradictions = [e for e in all_events if e.get("type") == "contradiction"]
        scars = [e for e in all_events if e.get("type") == "scar"]

        return {
            "total_contradictions": len(contradictions),
            "total_scars": len(scars),
            "quarantine_size": (
                os.path.getsize(self._quarantine_path)
                if self._quarantine_path.exists()
                else 0
            ),
            "chain_valid": self.ledger.verify_chain(),
        }