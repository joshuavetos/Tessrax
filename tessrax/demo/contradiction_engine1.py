import os
import json
import time
import hashlib
import threading
import traceback
import random
import logging
import types # Import types for patching
from pathlib import Path
from typing import Any, Dict, Callable, List, Optional
from functools import wraps

# We will keep the real nacl imports for now, but rely on the fact that we won't use them directly in this simplified demo init
# import nacl.signing # Import the real nacl for patching # No longer needed for simplified init
# from nacl.encoding import HexEncoder # No longer needed for simplified init
from prometheus_client import Counter, REGISTRY

# Configure basic logging (Ensure this is configured only once)
# Check if handlers already exist to avoid re-adding them in interactive environments
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Explicitly use In-Memory Patched/Successfully Imported Components ---
# Since importing the "real" components from the cloned repo is problematic,
# we will explicitly use the in-memory patched versions that were successfully
# defined in the ContradictionEngine cell (4bd680ff).

try:
    # Import the patched/in-memory components from __main__
    from __main__ import NonceRegistry, RevocationRegistry, verify_receipt, Tracer, trace, ResourceMonitor, ensure_in_sandbox
    # We still need a concrete ILedger implementation. Use the dummy one defined below.
    # We also need the ContradictionEngine class itself.
    from __main__ import ContradictionEngine

    logger.info("Successfully loaded in-memory patched components and ContradictionEngine.")

    # Define a Dummy ILedger as it's required by the ContradictionEngine init
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


except ImportError as e:
     logger.critical(f"Failed to load in-memory patched components or ContradictionEngine from __main__: {e}. Cannot run demo.", exc_info=True)
     # Define dummy placeholders if the imports from __main__ failed
     class NonceRegistry:
         def __init__(self): pass
         def register(self, *a, **kw): logger.error("Dummy NonceRegistry.register called.") ; return False
     class RevocationRegistry:
         def __init__(self): pass
         def revoke(self, *a, **kw): logger.error("Dummy RevocationRegistry.revoke called.") ; return None
     def verify_receipt(*a, **kw): logger.error("Dummy verify_receipt called."); return False
     class Tracer:
         def __init__(self, *a, **kw): logger.error("Dummy Tracer initialized."); self._active = False
         def record(self, *a, **kw): logger.error("Dummy Tracer.record called.")
         def flush(self): logger.error("Dummy Tracer.flush called.")
         def stop(self): logger.error("Dummy Tracer.stop called.")
         def trace(func):
             @wraps(func)
             def wrapper(*args, **kwargs): return func(*args, **kwargs)
             return wrapper
     class ResourceMonitor:
         def __init__(self, *a, **kw): logger.error("Dummy ResourceMonitor initialized.")
         def start(self): logger.error("Dummy ResourceMonitor.start called.")
         def stop(self): logger.error("Dummy ResourceMonitor.stop called.")
         def snapshot(self): logger.error("Dummy ResourceMonitor.snapshot called."); return {}
         def __enter__(self): return self
         def __exit__(self, exc_type, exc, tb): pass
     def ensure_in_sandbox(path, sandbox_root): logger.error("Dummy ensure_in_sandbox called."); return path
     class ILedger: # Dummy ILedger
          def __init__(self, *a, **kw): logger.error("Dummy ILedger initialized.")
          def add_event(self, event): logger.error("Dummy ILedger.add_event called.")
          def get_all_events(self, verify=False): logger.error("Dummy ILedger.get_all_events called."); return []
          def verify_chain(self): logger.error("Dummy ILedger.verify_chain called."); return False
     class ContradictionEngine: # Dummy ContradictionEngine
          def __init__(self, *a, **kw): logger.error("Dummy ContradictionEngine initialized.")
          def run_batch(self, events): logger.error("Dummy ContradictionEngine.run_batch called.")
          def get_stats(self): logger.error("Dummy ContradictionEngine.get_stats called."); return {}


# --- Temporary Patch for nacl.signing.SigningKey ---
# We will keep this patch in case any internal ContradictionEngine logic
# still tries to access nacl.signing.SigningKey, but the demo init will bypass
# the problematic generation and direct instantiation with hex.
try:
    OriginalSigningKey = nacl.signing.SigningKey

    class MockSigningKey:
        # Add the missing class attributes
        SEED_SIZE = 32 # Ed25519 seed size
        SIGNINGKEY_SIZE = 64 # Ed25519 signing key size
        VERIFYKEY_SIZE = 32 # Ed25519 verify key size
        SIGNATURE_SIZE = 64 # Ed25519 signature size


        def __init__(self, seed, encoder=HexEncoder):
            # We ignore the seed here and just provide a mock sign method
            logger.warning("[MOCK_NACL] Using MockSigningKey.")
            self.verify_key = MockVerifyKey() # Provide a mock verify_key

        def sign(self, message):
            logger.debug(f"[MOCK_NACL] MockSigningKey signing message: {message[:20]}...")
            # Return a dummy SignedMessage object that has a .signature attribute
            class MockSignedMessage:
                 def __init__(self, signature):
                      self.signature = signature
                 def hex(self):
                      # Need nacl.encoding.HexEncoder here, ensure it's imported if needed
                      try:
                          from nacl.encoding import HexEncoder as MockHexEncoder # Use alias to avoid conflict
                          return MockHexEncoder.encode(self.signature).decode()
                      except ImportError:
                           logger.error("HexEncoder not available for mock signature hex encoding.")
                           return "dummy_signature_hex"


            # Return a dummy signature (e.g., a hex string of appropriate length)
            dummy_signature_bytes = b'\x00' * MockSigningKey.SIGNATURE_SIZE
            return MockSignedMessage(dummy_signature_bytes)

        @staticmethod
        def generate():
            """Mock generate method for MockSigningKey."""
            logger.warning("[MOCK_NACL] Using MockSigningKey.generate().")
            # Return a dummy signing key instance
            # Use a consistent dummy seed/bytes for consistent "generation" in mock
            dummy_seed = b'\x00' * MockSigningKey.SEED_SIZE
            return MockSigningKey(dummy_seed)


    class MockVerifyKey:
         # Add the missing class attributes for direct access if needed,
         # or ensure access is via MockSigningKey.attribute
         VERIFYKEY_SIZE = MockSigningKey.VERIFYKEY_SIZE # Access from MockSigningKey
         SIGNATURE_SIZE = MockSigningKey.SIGNATURE_SIZE # Access from MockSigningKey


         def __init__(self):
              logger.warning("[MOCK_NACL] Using MockVerifyKey.")
         def encode(self, encoder):
              # Return a dummy public key hex string
              dummy_public_key_bytes = b'\x00' * self.VERIFYKEY_SIZE # Access using self or MockVerifyKey
              return encoder.encode(dummy_public_key_bytes)
         def verify(self, message, signature):
              logger.debug(f"[MOCK_NACL] MockVerifyKey verifying message: {message[:20]}...")
              # Always return True for mock verification
              return True

    # Patch the real SigningKey with our mock
    nacl.signing.SigningKey = MockSigningKey
    logger.info("[MOCK_NACL] Patched nacl.signing.SigningKey successfully.")

except ImportError as e:
     logger.warning(f"[MOCK_NACL] Could not import nacl.signing for patching: {e}. Skipping nacl patch.")
except Exception as e:
     logger.error(f"[MOCK_NACL] An unexpected error occurred during nacl patching: {e}", exc_info=True)


# -----------------------------------------------------


# Metrics (re-initialize or use existing if already defined)
try:
    CONTRADICTION_EVENTS_PROCESSED = Counter(
        "tessrax_contradictions_total",
        "Number of contradiction events processed by the engine"
    )
    logger.info("Prometheus Counter 'tessrax_contradictions_total' registered.")
except ValueError:
    try:
        CONTRADICTION_EVENTS_PROCESSED = REGISTRY._names_to_collectors["tessrax_contradictions_total"]
        logger.info("Prometheus Counter 'tessrax_contradictions_total' retrieved from registry.")
    except KeyError:
        logger.error("Prometheus Counter 'tessrax_contradictions_total' not found in registry after ValueError.")
        class DummyCounter:
            def inc(self, amount=1):
                logger.warning("DummyCounter 'inc' called as real counter not available.")
        CONTRADICTION_EVENTS_PROCESSED = DummyCounter()


# ------------------------------------------------------------
# Exceptions (Ensure base is defined first)
# ------------------------------------------------------------
class ContradictionEngineError(Exception):
    """Base exception for the Contradiction Engine."""
    pass

class QuarantineViolation(Exception):
    """Raised when an error occurs during quarantine operations."""
    pass

# Define custom exceptions for specific failure types
class VerificationError(ContradictionEngineError):
    """Raised when receipt verification fails."""
    pass

class RuleExecutionError(ContradictionEngineError):
    """Raised when an error occurs during rule execution."""
    pass

class LedgerInteractionError(ContradictionEngineError):
    """Raised when an error occurs interacting with the ledger."""
    pass


# ------------------------------------------------------------
# Contradiction Engine (Import the version with granular exception handling, logging, etc.)
# Assuming the improved ContradictionEngine class is available in the environment
# from a previous cell execution (cell 4bd680ff). If not, we would need to
# re-define it here or execute that cell first.
# For this step, we assume the class is already defined with improvements.
from __main__ import ContradictionEngine # Import the ContradictionEngine class from the cell it was defined in


# Retry decorator for transient errors (assuming it's defined in the ContradictionEngine cell)
# If not, define it here or ensure the ContradictionEngine cell is run first.
def retry(exceptions, tries=3, delay=1, backoff=2):
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"Retrying {f.__name__} after {type(e).__name__}: {e} - {mdelay:.2f}s delay, {mtries-1} tries left")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
                except Exception as e:
                    logger.error(f"Unexpected error during retry attempt for {f.__name__}: {type(e).__name__} - {e}", exc_info=True)
                    raise
            logger.info(f"Final attempt for {f.__name__}.")
            return f(*args, **kwargs)
        return f_retry
    return deco_retry


# 1. Define placeholder/mock implementations for the external dependencies.
#    These are now handled by importing from __main__ above or using dummy ILedger.

# 2. Define a simple example contradiction rule function.
def example_rule_value_mismatch(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Example rule: Detects a contradiction if 'input_value' does not match 'output_value'.
    """
    event_id = event.get('id', 'N/A')
    logger.debug(f"Executing example_rule_value_mismatch for event {event_id}.")
    inputs = event.get("inputs", {})
    outputs = event.get("outputs", {})

    input_value = inputs.get("input_value")
    output_value = outputs.get("output_value")

    # Check if both values exist and if they do not match
    if input_value is not None and output_value is not None and input_value != output_value:
        contradiction_payload = {
            "type": "value_mismatch",
            "event_id": event_id,
            "input_value": input_value,
            "output_value": output_value,
            "message": f"Input value ({input_value}) does not match output value ({output_value})."
        }
        logger.info(f"Rule detected contradiction for event {event_id}: Value mismatch.")
        return contradiction_payload
    else:
        logger.debug(f"No contradiction detected by example_rule_value_mismatch for event {event_id}.")
        return None

# 3. Define a simple example metabolism function.
def example_metabolize_contradiction(contradiction: Dict[str, Any]) -> None:
    """
    Example metabolism function: Prints a message when a contradiction is emitted.
    """
    contradiction_id = contradiction.get('id', 'N/A')
    contradiction_type = contradiction.get('type', 'unknown')
    logger.info(f"[METABOLISM] Metabolizing contradiction {contradiction_id} (Type: {contradiction_type}): {contradiction.get('message', 'No message')}")
    # In a real system, this might trigger alerts, store data in a separate DB, etc.

# 4. Initialize an instance of the ContradictionEngine using the real dependencies.
#    Completely bypass nacl key generation here and use a dummy hex string.
dummy_signing_key_hex_for_init = "00" * 32 # A dummy 32-byte hex string (Ed25519 seed size)
logger.info(f"Using hardcoded dummy signing key (hex) for engine: {dummy_signing_key_hex_for_init[:10]}...") # Log snippet


# Instantiate real dependencies (using patched/dummy versions from __main__)
real_ledger = ILedger() # Use the Dummy ILedger from __main__
real_nonce_registry = NonceRegistry() # Use the patched NonceRegistry from __main__
real_revocation_registry = RevocationRegistry() # Use the patched RevocationRegistry from __main__

# Initialize the engine with patched/dummy dependencies and the hardcoded key hex
engine = ContradictionEngine(
    ledger=real_ledger,
    ruleset=[example_rule_value_mismatch],
    signing_key_hex=dummy_signing_key_hex_for_init, # Pass the hardcoded hex string
    nonce_registry=real_nonce_registry,
    revocation_registry=real_revocation_registry,
    name="engine_demo", # Give the demo engine a specific name
    verify_strict=False, # Set to False for easier demonstration without full receipt structure
    quarantine_path="data/quarantine/engine_demo_quarantine.jsonl", # Use a specific quarantine file
    metabolize_fn=example_metabolize_contradiction
)
logger.info("ContradictionEngine instantiated with patched/dummy dependencies and hardcoded key hex.")


# 5. Create a few example event dictionaries.
example_events = [
    {
        "id": "event_001",
        "description": "Event with matching values",
        "inputs": {"input_value": 10},
        "outputs": {"output_value": 10},
        "receipt": {"dummy_receipt_data": "valid"} # Include a dummy receipt
    },
    {
        "id": "event_002",
        "description": "Event with value mismatch",
        "inputs": {"input_value": 100},
        "outputs": {"output_value": 200},
        "receipt": {"dummy_receipt_data": "valid"}
    },
    {
        "id": "event_003",
        "description": "Event missing output value",
        "inputs": {"input_value": 50},
        "outputs": {},
        "receipt": {"dummy_receipt_data": "valid"}
    },
     {
        "id": "event_004",
        "description": "Event missing input value",
        "inputs": {},
        "outputs": {"output_value": 75},
        "receipt": {"dummy_receipt_data": "valid"}
    },
    {
        "id": "event_005",
        "description": "Event with value mismatch and no receipt",
        "inputs": {"input_value": 99},
        "outputs": {"output_value": 1},
        # No receipt field
    },
     {
        "id": "event_006",
        "description": "Event with matching values and no receipt",
        "inputs": {"input_value": 5},
        "outputs": {"output_value": 5},
        # No receipt field
    },
]
logger.info(f"Created {len(example_events)} example events.")

# 6. Call the run_batch method with the list of example events.
logger.info("Running batch processing with example events.")
engine.run_batch(example_events)
logger.info("Batch processing complete.")

# 7. Call the get_stats method and print the returned statistics dictionary.
logger.info("Getting engine stats.")
stats = engine.get_stats()
print("\n--- Engine Statistics ---")
print(json.dumps(stats, indent=2))
print("-------------------------")

# Clean up the dummy quarantine file and ledger db for repeatable runs
if engine._quarantine_path.exists():
    try:
        os.remove(engine._quarantine_path)
        logger.info(f"Cleaned up dummy quarantine file: {engine._quarantine_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up dummy quarantine file: {e}")

# Clean up the SQLite ledger database file (even though we used a dummy Ledger, the path might be created)
ledger_db_path = Path("/content/data/engine_demo_ledger.db")
if ledger_db_path.exists():
    try:
        os.remove(ledger_db_path)
        logger.info(f"Cleaned up potential dummy ledger database file: {ledger_db_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up potential dummy ledger database file: {e}")


# 8. Briefly explain the example code.
print("\nExplanation:")
print("This code demonstrates the basic usage of the ContradictionEngine with patched/dummy Tessrax components.")
print("It initializes the engine with patched implementations of the ledger, registries, and tracer.")
print("A batch of example events is created, including some that should trigger the rule and some that shouldn't, and some missing receipts.")
print("Events that fail verification (because verify_strict is False and they lack receipts) or trigger the rule are handled.")
print("Events triggering the rule result in a contradiction being emitted (and metabolized by the example function).")
print("Finally, `get_stats` is called to show the engine's state after processing, including contradiction count and quarantine size.")
print("Note: This demo uses patched/dummy components due to import issues with the cloned repository, but demonstrates the ContradictionEngine's logic.")
print("The signing key used is for demonstration only and must be handled securely in production.")  # Execute the cell containing the engine demo code.
# This cell is tW2Km8kCbstO.
# It should now use the mock dependencies defined in cell c25b7bc7.

# The code in cell tW2Km8kCbstO already imports dependencies from __main__
# and defines the demo logic. We just need to execute it.
# No new code is needed here, just triggering the execution of the existing cell.  # --- Simple Mock Implementations for ContradictionEngine Dependencies ---

import logging
import time
import json
import threading
import random
from typing import Any, Dict, List, Optional
from pathlib import Path
from functools import wraps

# Configure basic logging (Ensure this is configured only once)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Mock ILedger
class MockLedger:
    """Simple mock ledger storing events in memory."""
    def __init__(self, db_path: str):
        logger.info(f"[MOCK_LEDGER] Initialized for DB path: {db_path}")
        self._events: List[Dict[str, Any]] = []

    def add_event(self, event: Dict[str, Any]) -> None:
        logger.info(f"[MOCK_LEDGER] Adding event: {event.get('type')}")
        self._events.append(event)
        # Simulate occasional transient errors for testing retry logic
        if random.random() < 0.1:
             logger.warning("[MOCK_LEDGER] Simulated transient error on add_event.")
             raise Exception("Simulated Ledger Error")


    def verify_chain(self) -> bool:
        logger.info("[MOCK_LEDGER] Verifying chain (mock always returns True)...")
        # Simulate occasional verification failure for testing
        if random.random() < 0.05:
            logger.warning("[MOCK_LEDGER] Simulated verification failure.")
            return False
        return True

    def get_all_events(self, verify: bool = False) -> List[Dict[str, Any]]:
        logger.info(f"[MOCK_LEDGER] Getting all events (verify={verify}).")
        # In a real ledger, 'verify' might trigger chain validation on retrieval.
        # For this mock, we just return the stored events.
        return self._events


# Mock NonceRegistry
class MockNonceRegistry:
    """Simple mock nonce registry storing nonces in a set."""
    def __init__(self):
        logger.info("[MOCK_NONCE_REGISTRY] Initialized.")
        self._nonces = set()
        self._lock = threading.Lock()

    def check_and_add(self, nonce: str, source: str) -> bool:
        """Checks if nonce exists and adds it if not. Returns True if new/valid, False if duplicate."""
        with self._lock:
            if nonce in self._nonces:
                logger.warning(f"[MOCK_NONCE_REGISTRY] Duplicate nonce detected: {nonce} from {source}")
                return False
            self._nonces.add(nonce)
            logger.debug(f"[MOCK_NONCE_REGISTRY] Registered nonce: {nonce} from {source}")
            return True


# Mock RevocationRegistry
class MockRevocationRegistry:
    """Simple mock revocation registry storing revoked cert IDs in a set."""
    def __init__(self):
        logger.info("[MOCK_REVOCATION_REGISTRY] Initialized.")
        self._revoked_certs = set()
        self._lock = threading.Lock()

    def revoke(self, cert_id: str) -> None:
        """Adds a certificate ID to the revoked list."""
        with self._lock:
            self._revoked_certs.add(cert_id)
            logger.info(f"[MOCK_REVOCATION_REGISTRY] Revoked certificate: {cert_id}")

    def is_revoked(self, cert_id: str) -> bool:
        """Checks if a certificate ID is in the revoked list."""
        with self._lock:
            is_revoked = cert_id in self._revoked_certs
            logger.debug(f"[MOCK_REVOCATION_REGISTRY] Checking revocation for {cert_id}: {is_revoked}")
            return is_revoked


# Mock Tracer
class MockTracer:
    """Minimal asynchronous tracer for runtime event logging."""

    def __init__(self, enable_async: bool = True, ledger: Optional[MockLedger] = None, private_key_hex: Optional[str] = None, executor_id: Optional[str] = None):
        logger.info(f"[MOCK_TRACER] Initialized for {executor_id} (async={enable_async}).")
        self.enable_async: bool = enable_async
        self._queue: List[Dict[str, Any]] = []
        self._lock: threading.Lock = threading.Lock()
        self._active: bool = True
        self.ledger: Optional[MockLedger] = ledger
        self.private_key_hex: Optional[str] = private_key_hex
        self.executor_id: Optional[str] = executor_id

        if self.enable_async:
            self._thread: threading.Thread = threading.Thread(target=self._flush_loop, daemon=True)
            self._thread.start()

    def record(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Record a trace event."""
        entry: Dict[str, Any] = {
            "timestamp": time.time(),
            "type": event_type,
            "payload": payload,
        }
        with self._lock:
            self._queue.append(entry)
            logger.debug(f"[MOCK_TRACER] Recorded event: {event_type}")


    def _flush_loop(self) -> None:
        """Asynchronous flushing loop."""
        logger.debug("[MOCK_TRACER] Flush loop started.")
        while self._active:
            self.flush()
            time.sleep(0.2)
        logger.debug("[MOCK_TRACER] Flush loop stopped.")


    def flush(self) -> None:
        """Print and clear all queued trace events."""
        with self._lock:
            while self._queue:
                e: Dict[str, Any] = self._queue.pop(0)
                logger.info(f"[MOCK_TRACER] Flushing event: {json.dumps(e)}")

    def stop(self) -> None:
        """Stop background thread and flush remaining events."""
        logger.info("[MOCK_TRACER] Stopping...")
        self._active = False
        if self.enable_async and hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)
        self.flush()


# Mock trace Decorator
def mock_trace(func: Callable) -> Callable:
    """Simple decorator for tracing function calls."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # logger.debug(f"[MOCK_TRACE_DECORATOR] Executing {func.__name__}") # Uncomment for more verbose trace
        return func(*args, **kwargs)
    return wrapper


# Mock ResourceMonitor
class MockResourceMonitor:
    """Simple mock resource monitor."""
    def __init__(self, name: str):
        logger.info(f"[MOCK_RESOURCE_MONITOR] Initialized {name}")

    def start(self) -> None:
        logger.debug("[MOCK_RESOURCE_MONITOR] Starting monitor.")

    def stop(self) -> None:
        logger.debug("[MOCK_RESOURCE_MONITOR] Stopping monitor.")

    def snapshot(self) -> Dict[str, Any]:
        logger.debug("[MOCK_RESOURCE_MONITOR] Taking snapshot.")
        return {"cpu": random.random(), "memory_mb": random.randint(100, 500), "timestamp": time.time()}

    def __enter__(self) -> "MockResourceMonitor":
        self.start()
        return self

    def __exit__(self, exc_type: Optional[type], exc: Optional[Exception], tb: Optional[traceback.TracebackException]) -> None:
        self.stop()
        return False


# Mock ensure_in_sandbox
def mock_ensure_in_sandbox(path: Path, sandbox_root: Path) -> Path:
    """Mocks ensuring a path is within the sandbox root."""
    logger.debug(f"[MOCK_SANDBOX] Ensuring {path} is in sandbox {sandbox_root}")
    # Simulate directory creation without strict sandbox checks
    target_path = sandbox_root / path.name # Simplify path handling for mock
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return target_path


# Mock verify_receipt (simplified)
def mock_verify_receipt(receipt: Dict[str, Any], strict: bool = True) -> bool:
    logger.debug(f"[MOCK_VERIFY_RECEIPT] Verifying receipt (strict={strict})...")
    # Simulate verification success/failure based on a simple check or randomness
    if not receipt:
        logger.warning("[MOCK_VERIFY_RECEIPT] No receipt provided, verification failed.")
        return False
    # For a simple mock, let's assume a receipt is valid if it has a 'dummy_receipt_data' key
    is_valid = "dummy_receipt_data" in receipt
    logger.debug(f"[MOCK_VERIFY_RECEIPT] Mock verification result: {is_valid}")
    return is_valid


# Define aliases to use these mocks with the ContradictionEngine expecting specific names
ILedger = MockLedger
NonceRegistry = MockNonceRegistry
RevocationRegistry = MockRevocationRegistry
Tracer = MockTracer
trace = mock_trace
ResourceMonitor = MockResourceMonitor
ensure_in_sandbox = mock_ensure_in_sandbox
verify_receipt = mock_verify_receipt
