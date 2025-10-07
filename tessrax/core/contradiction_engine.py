# Full code for the Contradiction Engine with necessary patches included

import os
import json
import time
import hashlib
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Callable, List, Optional

# Temporary patches for missing tessrax modules and components
import types, sys

# Patch for tessrax.core.receipts
receipts_patch = types.ModuleType("tessrax.core.receipts")

class NonceRegistry:
    def __init__(self): pass
    def register(self, *a, **kw): return True

class RevocationRegistry:
    def __init__(self): pass
    def revoke(self, *a, **kw): return True

def verify_receipt(*a, **kw):
    print("[WARN] Using dummy verify_receipt function.")
    return True # Assume valid for the patch

receipts_patch.NonceRegistry = NonceRegistry
receipts_patch.RevocationRegistry = RevocationRegistry
receipts_patch.verify_receipt = verify_receipt

sys.modules["tessrax.core.receipts"] = receipts_patch
# print("✅ Patched missing registries and verify_receipt function.") # Optional: uncomment for verification


# Patch for tessrax.utils.tracer
tracer_mod = types.ModuleType("tessrax.utils.tracer")

class Tracer:
    """
    Minimal replacement for the Tessrax Tracer class.
    Logs runtime trace events asynchronously into memory or print.
    """
    def __init__(self, enable_async: bool = True, ledger=None, private_key_hex=None, executor_id=None):
        # Added ledger, private_key_hex, executor_id to match ContradictionEngine init
        self.enable_async = enable_async
        self._queue = []
        self._lock = threading.Lock()
        self._active = True
        if self.enable_async:
            self._thread = threading.Thread(target=self._flush_loop, daemon=True)
            self._thread.start()
        # Dummy assignments to avoid errors if these attributes are accessed
        self.ledger = ledger
        self.private_key_hex = private_key_hex
        self.executor_id = executor_id


    def record(self, event_type: str, payload: dict):
        """Record a trace event."""
        entry = {
            "timestamp": time.time(),
            "type": event_type,
            "payload": payload,
        }
        with self._lock:
            self._queue.append(entry)

    def _flush_loop(self):
        """Asynchronous flusher."""
        while self._active:
            time.sleep(0.2)
            self.flush()

    def flush(self):
        """Print queued trace events."""
        with self._lock:
            while self._queue:
                e = self._queue.pop(0)
                print(f"[TRACE] {json.dumps(e)}")

    def stop(self):
        """Stop background flushing."""
        print("[WARN] Dummy Tracer stop called.")
        self._active = False
        if self.enable_async and hasattr(self, "_thread"):
            self._thread.join(timeout=1)

# Add a dummy trace decorator
def trace(func):
    def wrapper(*args, **kwargs):
        # print(f"[TRACE_DECORATOR] Calling {func.__name__}") # Optional: uncomment for decorator tracing
        return func(*args, **kwargs)
    return wrapper


utils_mod = types.ModuleType("tessrax.utils")
utils_mod.tracer = tracer_mod
tracer_mod.Tracer = Tracer
tracer_mod.trace = trace # Add the dummy decorator

sys.modules["tessrax.utils"] = utils_mod
sys.modules["tessrax.utils.tracer"] = tracer_mod

# print("✅ Patched Tracer module and decorator successfully.") # Optional: uncomment for verification


# Now, the original ContradictionEngine code

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

from nacl.signing import SigningKey
from nacl.encoding import HexEncoder
from prometheus_client import Counter, REGISTRY

from tessrax.core.interfaces import ILedger
# Import patched modules and components
from tessrax.core.receipts import verify_receipt, NonceRegistry, RevocationRegistry
from tessrax.core.resource_guard import ResourceMonitor, ensure_in_sandbox
from tessrax.utils.tracer import Tracer, trace


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
# Check if the metric already exists before creating it
try:
    CONTRADICTION_EVENTS_PROCESSED = REGISTRY.get_sample_value(
        "tessrax_contradictions_total"
    )
except KeyError:
    CONTRADICTION_EVENTS_PROCESSED: Counter = Counter(
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

        self.name: str = name
        self.ruleset: List[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = ruleset or []
        self.ledger: ILedger = ledger
        self.nonce_registry: NonceRegistry = nonce_registry
        self.revocation_registry: RevocationRegistry = revocation_registry
        self.verify_strict: bool = verify_strict
        self.metabolize_fn: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = metabolize_fn
        self._lock: threading.Lock = threading.Lock()

        self.signing_key: SigningKey = SigningKey(bytes.fromhex(signing_key_hex), encoder=HexEncoder)
        self.verify_key: str = self.signing_key.verify_key.encode(encoder=HexEncoder).decode()

        sandbox_root: Path = Path("data/sandbox")
        self._quarantine_path: Path = ensure_in_sandbox(Path(quarantine_path), sandbox_root)

        # Initialize runtime tracer (asynchronous, non-blocking)
        self.tracer: Tracer = Tracer(
            ledger=self.ledger,
            private_key_hex=signing_key_hex,  # Use the input hex here
            executor_id=self.name
        )

        self._running: bool = True
        self._monitor: ResourceMonitor = ResourceMonitor("ContradictionEngine")

    # --------------------------------------------------------
    # Core Methods
    # --------------------------------------------------------

    @trace # Apply the patched decorator
    def _sign_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Sign a payload using the engine's key."""
        serialized: bytes = json.dumps(payload, sort_keys=True).encode()
        signature: str = self.signing_key.sign(serialized).hex() # Use .hex() for consistency with verify_key
        payload["signature"] = signature
        payload["verify_key"] = self.verify_key
        return payload

    @trace # Apply the patched decorator
    def _verify_event(self, event: Dict[str, Any]) -> bool:
        """Verifies receipt validity."""
        try:
            # Use the patched verify_receipt
            return verify_receipt(event.get("receipt"), strict=self.verify_strict)
        except Exception as e:
            raise ContradictionEngineError(f"Receipt verification failed: {e}")

    @trace # Apply the patched decorator
    def _detect(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Runs all rules against event payload to detect contradictions."""
        contradictions: List[Dict[str, Any]] = []
        for rule in self.ruleset:
            try:
                result: Optional[Dict[str, Any]] = rule(event)
                if result:
                    contradictions.append(result)
            except Exception as e:
                print(f"[WARN] Rule {rule.__name__} raised error: {e}")
        return contradictions

    @trace # Apply the patched decorator
    def _quarantine(self, event: Dict[str, Any], reason: str) -> None:
        """Write quarantined event to durable forensic log."""
        try:
            record: Dict[str, Any] = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "reason": reason,
                "event": event,
            }
            # Ensure the data directory and sandbox exist
            self._quarantine_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._quarantine_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            raise QuarantineViolation(f"Failed to quarantine event: {e}")

    @trace # Apply the patched decorator
    def _emit(self, contradiction: Dict[str, Any]) -> None:
        """Sign and emit contradiction; ledger maintains its own chain."""
        base: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": "contradiction",
            "payload": contradiction,
        }
        signed: Dict[str, Any] = self._sign_payload(base)
        self.ledger.add_event(signed)
        if self.metabolize_fn:
            try:
                self.metabolize_fn(contradiction)
            except Exception as e:
                print(f"[WARN] Metabolism failed: {e}")
        # Increment the metric using the retrieved or newly created Counter object
        if isinstance(CONTRADICTION_EVENTS_PROCESSED, Counter):
            CONTRADICTION_EVENTS_PROCESSED.inc()
        else:
             # Handle the case where get_sample_value returned a scalar (shouldn't happen for Counter)
             print("[WARN] Metric increment skipped: CONTRADICTION_EVENTS_PROCESSED is not a Counter instance.")


    # --------------------------------------------------------
    # Batch / Loop
    # --------------------------------------------------------

    def _run_once_unlocked(self, event: Dict[str, Any]) -> None:
        """Runs the full pipeline for a single event."""
        if not self._verify_event(event):
            self._quarantine(event, "Failed receipt verification")
            return

        contradictions: List[Dict[str, Any]] = self._detect(event)
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
                event: Dict[str, Any] = event_source()
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
        # Check if tracer exists before stopping
        if hasattr(self, 'tracer') and self.tracer:
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
        all_events: List[Dict[str, Any]] = self.ledger.get_all_events(verify=False)
        contradictions: List[Dict[str, Any]] = [e for e in all_events if e.get("type") == "contradiction"]
        scars: List[Dict[str, Any]] = [e for e in all_events if e.get("type") == "scar"]
        return {
            "total_contradictions": len(contradictions),
            "total_scars": len(scars),
            "quarantine_size": os.path.getsize(self._quarantine_path)
            if self._quarantine_path.exists()
            else 0,
            "chain_valid": self.ledger.verify_chain(),
        }
