"""
Tessrax Contradiction Engine v4.9 (Hardened + Traced)
-----------------------------------------------------
Secure contradiction-processing runtime:
✓ Verified receipts (secure-by-default)
✓ Sandboxed, durable quarantine with fail-safe handling
✓ Signed contradiction records (delegates chaining to ledger)
✓ Runtime tracer integration (auditable heartbeat)
✓ Modular ledger contract (ILedger-compliant)
"""

import os
import json
import time
import hashlib
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Callable, List, Optional

from nacl.signing import SigningKey
from nacl.encoding import HexEncoder
from prometheus_client import Counter

from tessrax.core.interfaces import ILedger
from tessrax.core.receipts import verify_receipt, NonceRegistry, RevocationRegistry
from tessrax.core.resource_guard import ResourceMonitor, ensure_in_sandbox
from tessrax.utils.tracer import Tracer


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
CONTRADICTION_EVENTS_PROCESSED = Counter(
    "tessrax_contradictions_total",
    "Number of contradiction events processed by the engine"
)


# ------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------
class ContradictionEngineError(Exception):
    pass


class QuarantineViolation(Exception):
    pass


# ------------------------------------------------------------
# Contradiction Engine
# ------------------------------------------------------------
class ContradictionEngine:
    """
    Hardened Contradiction Engine with ledger, tracer, and audit integration.
    """

    def __init__(
        self,
        *,
        ledger: ILedger,
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

        # Initialize runtime tracer (asynchronous, non-blocking)
        self.tracer = Tracer(
            ledger=self.ledger,
            private_key_hex=self.signing_key_hex,
            executor_id=self.name
        )

        self._running = True
        self._monitor = ResourceMonitor("ContradictionEngine")

    # --------------------------------------------------------
    # Core Methods
    # --------------------------------------------------------

    def _sign_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Sign a payload using the engine's key."""
        serialized = json.dumps(payload, sort_keys=True).encode()
        signature = self.signing_key.sign(serialized).signature.hex()
        payload["signature"] = signature
        payload["verify_key"] = self.verify_key
        return payload

    @Tracer.trace
    def _verify_event(self, event: Dict[str, Any]) -> bool:
        """Verifies receipt validity."""
        try:
            return verify_receipt(event.get("receipt"), strict=self.verify_strict)
        except Exception as e:
            raise ContradictionEngineError(f"Receipt verification failed: {e}")

    @Tracer.trace
    def _detect(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Runs all rules against event payload to detect contradictions."""
        contradictions = []
        for rule in self.ruleset:
            try:
                result = rule(event)
                if result:
                    contradictions.append(result)
            except Exception as e:
                print(f"[WARN] Rule {rule.__name__} raised error: {e}")
        return contradictions

    @Tracer.trace
    def _quarantine(self, event: Dict[str, Any], reason: str) -> None:
        """Write quarantined event to durable forensic log."""
        try:
            record = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "reason": reason,
                "event": event,
            }
            with open(self._quarantine_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            raise QuarantineViolation(f"Failed to quarantine event: {e}")

    @Tracer.trace
    def _emit(self, contradiction: Dict[str, Any]) -> None:
        """Sign and emit contradiction; ledger maintains its own chain."""
        base = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": "contradiction",
            "payload": contradiction,
        }
        signed = self._sign_payload(base)
        self.ledger.add_event(signed)
        if self.metabolize_fn:
            try:
                self.metabolize_fn(contradiction)
            except Exception as e:
                print(f"[WARN] Metabolism failed: {e}")
        CONTRADICTION_EVENTS_PROCESSED.inc()

    # --------------------------------------------------------
    # Batch / Loop
    # --------------------------------------------------------

    def _run_once_unlocked(self, event: Dict[str, Any]) -> None:
        """Runs the full pipeline for a single event."""
        if not self._verify_event(event):
            self._quarantine(event, "Failed receipt verification")
            return

        contradictions = self._detect(event)
        for c in contradictions:
            self._emit(c)

    def run_batch(self, events: List[Dict[str, Any]]) -> None:
        """Processes multiple events with quarantine fault tolerance."""
        with self._lock:
            for ev in events:
                try:
                    self._run_once_unlocked(ev)
                except QuarantineViolation as e:
                    print(f"[CRITICAL] Quarantine write failure: {e}")
                    self.stop()
                    break

    def run_forever(self, event_source: Callable[[], Dict[str, Any]], delay: float = 1.0):
        """Continuously polls event source and processes in real time."""
        print(f"[INFO] {self.name} running...")
        while self._running:
            try:
                event = event_source()
                if event:
                    self.run_batch([event])
                time.sleep(delay)
            except KeyboardInterrupt:
                self.stop()
            except Exception as e:
                print(f"[ERROR] Loop exception: {e}")
                self._quarantine({"error": str(e)}, "Runtime failure")

    def stop(self):
        """Stops the main loop and shuts down tracer thread."""
        print(f"[INFO] {self.name} stopping...")
        self._running = False
        self.tracer.stop()

    # --------------------------------------------------------
    # Verification / Stats
    # --------------------------------------------------------

    def verify_contradiction_chain(self) -> bool:
        """Delegates full verification to the ledger."""
        if not hasattr(self.ledger, "verify_chain"):
            raise ContradictionEngineError("Ledger missing `verify_chain()` method.")
        return self.ledger.verify_chain()

    def get_stats(self) -> Dict[str, Any]:
        """Reports engine health and metrics."""
        all_events = self.ledger.get_all_events(verify=False)
        contradictions = [e for e in all_events if e.get("type") == "contradiction"]
        scars = [e for e in all_events if e.get("type") == "scar"]
        return {
            "total_contradictions": len(contradictions),
            "total_scars": len(scars),
            "quarantine_size": os.path.getsize(self._quarantine_path)
            if self._quarantine_path.exists()
            else 0,
            "chain_valid": self.ledger.verify_chain(),
        }