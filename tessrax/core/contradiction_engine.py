# Full code for the Contradiction Engine with necessary patches included

import os
import json
import time
import hashlib
import threading
import traceback
import types, sys # Import types and sys for patching
import logging # Import the logging module
from pathlib import Path
from typing import Any, Dict, Callable, List, Optional

from nacl.signing import SigningKey
from nacl.encoding import HexEncoder
from prometheus_client import Counter, REGISTRY # Import REGISTRY to check for existing metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Temporary patches for missing tessrax modules and components

# Patch for tessrax.core.interfaces (assuming ILedger is the only dependency needed from here for the class definition)
# If other interfaces are needed later, this patch might need expansion.
interfaces_patch = types.ModuleType("tessrax.core.interfaces")
class ILedger:
    """Dummy ILedger interface."""
    def __init__(self, *args: Any, **kwargs: Any):
        logger.warning("Using dummy ILedger.")
    def add_event(self, event: Dict[str, Any]) -> None:
        logger.warning(f"Dummy add_event called with: {event}")
    def get_all_events(self, verify: bool = False) -> List[Dict[str, Any]]:
        logger.warning("Dummy get_all_events called.")
        return [] # Return empty list or dummy data as needed
    def verify_chain(self) -> bool:
        logger.warning("Dummy verify_chain called.")
        return True # Assume valid for dummy

sys.modules["tessrax.core.interfaces"] = interfaces_patch
interfaces_patch.ILedger = ILedger
# logger.info("✅ Patched tessrax.core.interfaces successfully.") # Optional: uncomment for verification


# Patch for tessrax.core.receipts
receipts_patch = types.ModuleType("tessrax.core.receipts")

class NonceRegistry:
    def __init__(self): pass
    def register(self, *a: Any, **kw: Any) -> bool: return True

class RevocationRegistry:
    def __init__(self): pass
    def revoke(self, *a: Any, **kw: Any) -> None: return None

def verify_receipt(*a: Any, **kw: Any) -> bool:
    logger.warning("Using dummy verify_receipt function.")
    return True # Assume valid for the patch

receipts_patch.NonceRegistry = NonceRegistry
receipts_patch.RevocationRegistry = RevocationRegistry
receipts_patch.verify_receipt = verify_receipt

sys.modules["tessrax.core.receipts"] = receipts_patch
logger.info("✅ Patched missing registries and verify_receipt function.")


# Patch for tessrax.core.resource_guard
resource_guard_patch = types.ModuleType("tessrax.core.resource_guard")

class ResourceMonitor:
    """Dummy ResourceMonitor."""
    def __init__(self, *args: Any, **kwargs: Any):
        logger.warning("Using dummy ResourceMonitor.")
        self._active: bool = False # Add type hint for _active
        self._thread: Optional[threading.Thread] = None # Add type hint for _thread
        self._violation: Optional[str] = None # Add type hint for _violation

    def start(self) -> None:
        logger.warning("Dummy ResourceMonitor start called.")
    def stop(self) -> None:
        logger.warning("Dummy ResourceMonitor stop called.")
    def snapshot(self) -> Dict[str, Any]:
        logger.warning("Dummy ResourceMonitor snapshot called.")
        return {"cpu": 0.0, "memory_mb": 0.0, "timestamp": time.time()}
    def __enter__(self) -> "ResourceMonitor": return self
    def __exit__(self, exc_type: Optional[type], exc: Optional[Exception], tb: Optional[traceback.TracebackException]) -> None: pass


def ensure_in_sandbox(path: Path, sandbox_root: Path) -> Path:
    logger.warning(f"Using dummy ensure_in_sandbox for path: {path}")
    # In a real scenario, this would enforce sandbox rules.
    # For the patch, we just return the path and ensure parent directory exists for quarantine.
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


resource_guard_patch.ResourceMonitor = ResourceMonitor
resource_guard_patch.ensure_in_sandbox = ensure_in_sandbox
sys.modules["tessrax.core.resource_guard"] = resource_guard_patch
# logger.info("✅ Patched tessrax.core.resource_guard successfully.") # Optional: uncomment for verification


# Patch for tessrax.utils.tracer
tracer_mod = types.ModuleType("tessrax.utils.tracer")

class Tracer:
    """
    Minimal replacement for the Tessrax Tracer class.
    Logs runtime trace events asynchronously into memory or print.
    """
    def __init__(self, enable_async: bool = True, ledger: Optional[ILedger] = None, private_key_hex: Optional[str] = None, executor_id: Optional[str] = None):
        # Added ledger, private_key_hex, executor_id to match ContradictionEngine init
        self.enable_async: bool = enable_async
        self._queue: List[Dict[str, Any]] = []
        self._lock: threading.Lock = threading.Lock()
        self._active: bool = True
        if self.enable_async:
            self._thread: threading.Thread = threading.Thread(target=self._flush_loop, daemon=True)
            self._thread.start()
        # Dummy assignments to avoid errors if these attributes are accessed
        self.ledger: Optional[ILedger] = ledger
        self.private_key_hex: Optional[str] = private_key_hex
        self.executor_id: Optional[str] = executor_id


    def record(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Record a trace event."""
        entry: Dict[str, Any] = {
            "timestamp": time.time(),
            "type": event_type,
            "payload": payload,
        }
        with self._lock:
            self._queue.append(entry)

    def _flush_loop(self) -> None:
        """Asynchronous flusher."""
        while self._active:
            time.sleep(0.2)
            self.flush()

    def flush(self) -> None:
        """Print queued trace events."""
        with self._lock:
            while self._queue:
                e: Dict[str, Any] = self._queue.pop(0)
                logger.info(f"[TRACE] {json.dumps(e)}") # Use logger

    def stop(self) -> None:
        """Stop background flushing."""
        logger.warning("[WARN] Dummy Tracer stop called.") # Use logger
        self._active = False
        if self.enable_async and hasattr(self, "_thread"):
            self._thread.join(timeout=1)

# Add a dummy trace decorator
def trace(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # logger.info(f"[TRACE_DECORATOR] Calling {func.__name__}") # Optional: uncomment for decorator tracing
        return func(*args, **kwargs)
    return wrapper


utils_mod = types.ModuleType("tessrax.utils")
utils_mod.tracer = tracer_mod
tracer_mod.Tracer = Tracer
tracer_mod.trace = trace # Add the dummy decorator

sys.modules["tessrax.utils"] = utils_mod
sys.modules["tessrax.utils.tracer"] = tracer_mod

# logger.info("✅ Patched Tracer module and decorator successfully.") # Optional: uncomment for verification


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

        # Make sandbox_root configurable, default to "data/sandbox"
        sandbox_root_str: str = os.environ.get("TESSRAX_SANDBOX_ROOT", "data/sandbox")
        sandbox_root: Path = Path(sandbox_root_str)

        # Make quarantine_path configurable, default to "quarantine.jsonl" within sandbox
        quarantine_filename: str = os.environ.get("TESSRAX_QUARANTINE_FILENAME", "quarantine.jsonl")
        self._quarantine_path: Path = ensure_in_sandbox(sandbox_root / quarantine_filename, sandbox_root)


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
        try:
            signature: str = self.signing_key.sign(serialized).hex() # Use .hex() for consistency with verify_key
            payload["signature"] = signature
            payload["verify_key"] = self.verify_key
            return payload
        except Exception as e:
             logger.error(f"Error signing payload: {e}", exc_info=True)
             raise ContradictionEngineError(f"Error signing payload: {e}")


    @trace # Apply the patched decorator
    def _verify_event(self, event: Dict[str, Any]) -> bool:
        """Verifies receipt validity."""
        try:
            # Use the patched verify_receipt
            return verify_receipt(event.get("receipt"), strict=self.verify_strict)
        except Exception as e:
            logger.error(f"Receipt verification failed: {e}", exc_info=True) # Use logger
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
                logger.warning(f"Rule {rule.__name__} raised error: {e}", exc_info=True) # Use logger
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
        except IOError as e: # Catch specific IOError for file operations
            logger.critical(f"Failed to write to quarantine file: {e}", exc_info=True) # Use logger
            raise QuarantineViolation(f"Failed to quarantine event: {e}")
        except Exception as e: # Catch other potential exceptions during JSON dump etc.
            logger.critical(f"Failed to quarantine event (other error): {e}", exc_info=True) # Use logger
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
        try:
            self.ledger.add_event(signed)
        except Exception as e: # Catch exceptions from ledger interaction
             logger.error(f"Error adding event to ledger: {e}", exc_info=True)
             # Depending on desired behavior, you might re-raise or handle differently
             raise ContradictionEngineError(f"Error adding event to ledger: {e}")


        if self.metabolize_fn:
            try:
                self.metabolize_fn(contradiction)
            except Exception as e:
                logger.warning(f"Metabolism failed: {e}", exc_info=True) # Use logger
        # Increment the metric using the retrieved or newly created Counter object
        if isinstance(CONTRADICTION_EVENTS_PROCESSED, Counter):
            CONTRADICTION_EVENTS_PROCESSED.inc()
        else:
             # Handle the case where get_sample_value returned a scalar (shouldn't happen for Counter)
             logger.warning("Metric increment skipped: CONTRADICTION_EVENTS_PROCESSED is not a Counter instance.")


    # --------------------------------------------------------
    # Batch / Loop
    # --------------------------------------------------------

    def _run_once_unlocked(self, event: Dict[str, Any]) -> None:
        """Runs the full pipeline for a single event."""
        if not self._verify_event(event):
            self._quarantine(event, "Failed receipt verification")
            return

        try:
            contradictions: List[Dict[str, Any]] = self._detect(event)
            for c in contradictions:
                self._emit(c)
        except Exception as e: # Catch exceptions during detection or emission
             logger.error(f"Error during event processing pipeline: {e}", exc_info=True)
             self._quarantine(event, f"Processing pipeline error: {e}")


    def run_batch(self, events: List[Dict[str, Any]]) -> None:
        """Processes multiple events with quarantine fault tolerance."""
        with self._lock:
            for ev in events:
                try:
                    self._run_once_unlocked(ev)
                except QuarantineViolation as e:
                    logger.critical(f"Quarantine write failure: {e}", exc_info=True) # Use logger
                    self.stop()
                    break
                except Exception as e: # Catch any other unexpected errors during batch processing
                    logger.error(f"Unexpected error processing event in batch: {e}", exc_info=True)
                    self._quarantine(ev, f"Unexpected batch processing error: {e}")


    def run_forever(self, event_source: Callable[[], Dict[str, Any]], delay: float = 1.0):
        """Continuously polls event source and processes in real time."""
        logger.info(f"{self.name} running...") # Use logger
        while self._running:
            try:
                event: Dict[str, Any] = event_source()
                if event:
                    self.run_batch([event])
                time.sleep(delay)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Stopping.") # Use logger
                self.stop()
            except Exception as e:
                logger.error(f"Loop exception: {e}", exc_info=True) # Use logger
                # Decide if you want to quarantine the error event itself or the last processed event
                self._quarantine({"error": str(e)}, "Runtime failure")


    def stop(self):
        """Stops the main loop and shuts down tracer thread."""
        logger.info(f"{self.name} stopping...") # Use logger
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
        try:
            return self.ledger.verify_chain()
        except Exception as e:
             logger.error(f"Error verifying contradiction chain: {e}", exc_info=True)
             raise ContradictionEngineError(f"Error verifying contradiction chain: {e}")


    def get_stats(self) -> Dict[str, Any]:
        """Reports engine health and metrics."""
        try:
            all_events: List[Dict[str, Any]] = self.ledger.get_all_events(verify=False)
            contradictions: List[Dict[str, Any]] = [e for e in all_events if e.get("type") == "contradiction"]
            scars: List[Dict[str, Any]] = [e for e in all_events if e.get("type") == "scar"]
            quarantine_size = os.path.getsize(self._quarantine_path) if self._quarantine_path.exists() else 0
            chain_valid = self.ledger.verify_chain() # This might raise an exception, already handled in verify_contradiction_chain

            return {
                "total_contradictions": len(contradictions),
                "total_scars": len(scars),
                "quarantine_size": quarantine_size,
                "chain_valid": chain_valid,
            }
        except Exception as e:
            logger.error(f"Error getting engine stats: {e}", exc_info=True)
            raise ContradictionEngineError(f"Error getting engine stats: {e}")
