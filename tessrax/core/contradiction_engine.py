# tessrax/core/contradiction_engine.py
"""
Tessrax Contradiction Engine v5.0 — Hardened + Auditable Heartbeat
-------------------------------------------------------------------
Secure contradiction-processing runtime with provenance tracing.

Features:
✓ Verified receipts (secure-by-default)
✓ Sandboxed, durable quarantine with fail-safe handling
✓ Signed contradiction records (delegates chaining to ledger)
✓ Optional scar metabolism hooks
✓ Rule registration and robust error handling
✓ Integrated asynchronous tracer ("auditable heartbeat")
"""

import os
import json
import time
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

# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------
CONTRADICTION_EVENTS_PROCESSED = Counter(
    "tessrax_contradiction_events_processed_total",
    "Total number of contradiction events successfully processed."
)

# -------------------------------------------------------------------
# Exceptions
# -------------------------------------------------------------------
class ContradictionEngineError(Exception):
    """Base class for contradiction engine errors."""


class QuarantineViolation(ContradictionEngineError):
    """Raised when a critical quarantine operation fails."""


# -------------------------------------------------------------------
# Core Engine
# -------------------------------------------------------------------
class ContradictionEngine:
    """
    Hardened Contradiction Engine with full runtime provenance tracing.
    """

    def __init__(
        self,
        *,
        ledger: ILedger,
        signing_key_hex: str,
        nonce_registry: NonceRegistry,
        revocation_registry: RevocationRegistry,
        ruleset: Optional[List[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]]] = None,
        name: str = "contradiction_engine",
        verify_strict: bool = True,
        quarantine_path: str = "data/quarantine.jsonl",
        metabolize_fn: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    ):
        if not all([ledger, signing_key_hex, nonce_registry, revocation_registry]):
            raise ContradictionEngineError(
                "Ledger, signing_key_hex, NonceRegistry, and RevocationRegistry are required."
            )

        self.name = name
        self.ledger = ledger
        self.ruleset = ruleset or []
        self.nonce_registry = nonce_registry
        self.revocation_registry = revocation_registry
        self.verify_strict = verify_strict
        self.metabolize_fn = metabolize_fn
        self._lock = threading.Lock()

        self.signing_key = SigningKey(signing_key_hex, encoder=HexEncoder)
        self.verify_key = self.signing_key.verify_key.encode(encoder=HexEncoder).decode()

        sandbox_root = Path("data/sandbox")
        self._quarantine_path = ensure_in_sandbox(Path(quarantine_path), sandbox_root)

        # Initialize tracer (asynchronous, non-blocking provenance layer)
        self.tracer = Tracer(
            ledger=self.ledger,
            private_key_hex=signing_key_hex,
            executor_id=self.name,
            sample_rate=1.0
        )

    # ----------------------------------------------------------------
    # Traced internal operations (auditable heartbeat)
    # ----------------------------------------------------------------
    @property
    def tracer_trace(self):
        """Helper to use self.tracer.trace cleanly with bound methods."""
        return self.tracer.trace

    @tracer_trace
    def _verify_event(self, event: Dict[str, Any]) -> bool:
        """Verifies an incoming event's authenticity."""
        if not event or "receipt" not in event:
            raise ContradictionEngineError("Event missing required receipt.")
        return verify_receipt(event["receipt"], strict=self.verify_strict)

    @tracer_trace
    def _quarantine(self, event: Dict[str, Any], reason: str) -> None:
        """Writes an event to the sandboxed quarantine log."""
        try:
            with open(self._quarantine_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"reason": reason, "event": event}) + "\n")
        except Exception as e:
            raise QuarantineViolation(f"Failed to write to quarantine: {e}")

    @tracer_trace
    def _detect(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Applies the rule set to detect contradictions in the event."""
        contradictions = []
        for rule in self.ruleset:
            try:
                result = rule(event)
                if result:
                    contradictions.append(result)
            except Exception as e:
                print(f"[Rule Error] {rule.__name__}: {e}")
        return contradictions

    @tracer_trace
    def _emit(self, contradiction: Dict[str, Any]) -> None:
        """Signs and emits a contradiction record to the ledger."""
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": "contradiction",
            "payload": contradiction,
        }
        signed = self._sign_payload(record)
        self.ledger.add_event(signed)

        if self.metabolize_fn:
            try:
                self.metabolize_fn(contradiction)
            except Exception as e:
                print(f"[WARN] Metabolism function failed: {e}")

        CONTRADICTION_EVENTS_PROCESSED.inc()

    # ----------------------------------------------------------------
    # Main control flow
    # ----------------------------------------------------------------
    def run_batch(self, events: List[Dict[str, Any]]) -> None:
        """Processes a batch of events safely and audibly."""
        with self._lock:
            for ev in events:
                try:
                    if self._verify_event(ev):
                        contradictions = self._detect(ev)
                        for c in contradictions:
                            self._emit(c)
                except QuarantineViolation as qv:
                    print(f"[CRITICAL] Quarantine failure: {qv}")
                    self.stop()
                    break
                except Exception as e:
                    self._quarantine(ev, f"Unhandled processing error: {e}")

    # ----------------------------------------------------------------
    # Utility and reporting
    # ----------------------------------------------------------------
    def _sign_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Signs payload using engine's private key."""
        message = json.dumps(payload, sort_keys=True).encode()
        signature = self.signing_key.sign(message).signature.hex()
        payload_hash = hashlib.sha256(message).hexdigest()
        return {
            "payload": payload,
            "signature": signature,
            "verify_key": self.verify_key,
            "hash": payload_hash,
        }

    def verify_contradiction_chain(self) -> bool:
        """Delegates full-chain verification to the ledger."""
        if not hasattr(self.ledger, "verify_chain"):
            raise ContradictionEngineError("Ledger missing `verify_chain()` method.")
        return self.ledger.verify_chain()

    def get_stats(self) -> Dict[str, Any]:
        """Returns runtime statistics."""
        all_events = self.ledger.get_all_events(verify=False)
        contradictions = [e for e in all_events if e.get("type") == "contradiction"]
        scars = [e for e in all_events if e.get("type") == "scar"]
        quarantine_size = (
            os.path.getsize(self._quarantine_path)
            if self._quarantine_path.exists()
            else 0
        )
        return {
            "total_contradictions": len(contradictions),
            "total_scars": len(scars),
            "quarantine_size": quarantine_size,
            "chain_valid": self.ledger.verify_chain(),
        }

    def stop(self):
        """Gracefully stops background processes (e.g., tracer)."""
        try:
            self.tracer.stop()
        except Exception:
            pass