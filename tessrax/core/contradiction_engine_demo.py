import os
import json
import time
import hashlib
import threading
import traceback
import random
import logging
from pathlib import Path
from typing import Any, Dict, Callable, List, Optional
from functools import wraps

from nacl.signing import SigningKey
from nacl.encoding import HexEncoder
from prometheus_client import Counter, REGISTRY

# Configure basic logging (Ensure this is configured only once)
# Check if handlers already exist to avoid re-adding them in interactive environments
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Mock/Placeholder classes and functions for demonstration ---
# In a real scenario, these would be imported from tessrax
# These are included here to make the ContradictionEngine class runnable
# in isolation for demonstration and testing purposes.
class ILedger:
    """Mock Ledger interface."""
    def add_event(self, event):
        """Adds an event to the ledger."""
        logger.info(f"[MOCK_LEDGER] Adding event: {event.get('type')}")
        # Simulate a transient error occasionally
        if random.random() < 0.1:
            logger.warning("[MOCK_LEDGER] Simulated transient error on add_event.")
            raise Exception("Simulated Ledger Error")
        pass

    def verify_chain(self):
        """Verifies the integrity of the ledger chain."""
        logger.info("[MOCK_LEDGER] Verifying chain...")
        # Simulate occasional verification failure
        if random.random() < 0.05:
            logger.warning("[MOCK_LEDGER] Simulated verification failure.")
            return False
        return True

    def get_all_events(self, verify=False):
        """Retrieves all events from the ledger."""
        logger.info("[MOCK_LEDGER] Getting all events.")
        # Return some mock events
        return [{"type": "contradiction", "id": "mock_c_1"}, {"type": "scar", "id": "mock_s_1"}]

class NonceRegistry:
    """Mock Nonce Registry for receipt replay protection."""
    def check_and_add(self, nonce, source):
        """Checks if a nonce has been used and adds it."""
        logger.debug(f"[MOCK_NONCE_REGISTRY] Checking/adding nonce {nonce} from {source}")
        return True # Always valid for mock

class RevocationRegistry:
    """Mock Revocation Registry for revoked certificates."""
    def is_revoked(self, cert_id):
        """Checks if a certificate ID is revoked."""
        logger.debug(f"[MOCK_REVOCATION_REGISTRY] Checking revocation for {cert_id}")
        return False # Never revoked for mock

class ResourceMonitor:
    """Mock Resource Monitor."""
    def __init__(self, name):
        logger.info(f"[MOCK_RESOURCE_MONITOR] Initialized {name}")
    def __enter__(self):
        logger.debug("[MOCK_RESOURCE_MONITOR] Starting monitor.")
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.debug("[MOCK_RESOURCE_MONITOR] Stopping monitor.")

def ensure_in_sandbox(path: Path, sandbox_root: Path) -> Path:
    """Mocks ensuring a path is within the sandbox root."""
    logger.debug(f"[MOCK_SANDBOX] Ensuring {path} is in sandbox {sandbox_root}")
    # Simulate creating the sandbox directory
    sandbox_root.mkdir(parents=True, exist_ok=True)
    # Simulate returning a path within the sandbox
    return sandbox_root / path.name

class Tracer:
    """Mock Tracer for runtime auditing."""
    def __init__(self, ledger, private_key_hex, executor_id):
        logger.info(f"[MOCK_TRACER] Initialized for {executor_id}")
        self._running = True
        # In a real implementation, this would likely start a thread
        # self._thread = threading.Thread(target=self._run)
        # self._thread.daemon = True
        # self._thread.start()
        print("[MOCK_TRACER] Initialized (thread not started in mock).")


    def _run(self):
        """Simulates tracing activity."""
        logger.debug("[MOCK_TRACER] Running...")
        while self._running:
            time.sleep(5) # Simulate tracing work
            logger.debug("[MOCK_TRACER] Tracing...")
        logger.debug("[MOCK_TRACER] Stopped.")

    def stop(self):
        """Stops the tracer."""
        logger.info("[MOCK_TRACER] Stopping...")
        self._running = False
        # If using a thread, join it here
        # if self._thread.is_alive():
        #     self._thread.join(timeout=1)
        print("[MOCK_TRACER] Stop requested.")


    @staticmethod
    def trace(func: Callable) -> Callable:
        """Decorator to simulate tracing function calls."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # logger.debug(f"[MOCK_TRACER] Tracing call to {func.__name__}")
            return func(*args, **kwargs)
        return wrapper

def verify_receipt(receipt: Dict[str, Any], strict: bool = True) -> bool:
    """Mocks verification of an event receipt."""
    logger.debug(f"[MOCK_VERIFY_RECEIPT] Verifying receipt (strict={strict})...")
    # Simulate occasional verification failure
    if random.random() < 0.02:
        logger.warning("[MOCK_VERIFY_RECEIPT] Simulated verification failure.")
        return False
    # Simulate occasional error during verification
    if random.random() < 0.01:
        logger.error("[MOCK_VERIFY_RECEIPT] Simulated error during verification.")
        raise Exception("Simulated Verification Error")
    return True


# Metrics (re-initialize or use existing if already defined)
# Ensure this doesn't re-register if already run.
# This pattern is common in interactive environments like notebooks.
try:
    CONTRADICTION_EVENTS_PROCESSED = Counter(
        "tessrax_contradictions_total",
        "Number of contradiction events processed by the engine"
    )
    logger.info("Prometheus Counter 'tessrax_contradictions_total' registered.")
except ValueError:
    # Counter already registered, try to get it from the registry
    try:
        CONTRADICTION_EVENTS_PROCESSED = REGISTRY._names_to_collectors["tessrax_contradictions_total"]
        logger.info("Prometheus Counter 'tessrax_contradictions_total' retrieved from registry.")
    except KeyError:
        logger.error("Prometheus Counter 'tessrax_contradictions_total' not found in registry after ValueError.")
        # Fallback to a dummy object if retrieving fails, to prevent runtime errors
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
# Contradiction Engine
# ------------------------------------------------------------

# Retry decorator for transient errors
def retry(exceptions, tries=3, delay=1, backoff=2):
    """
    Retry calling the decorated function using an exponential backoff.

    Args:
        exceptions: The exception(s) to catch and retry on. Can be a single
                    exception type or a tuple of exception types.
        tries: Maximum number of attempts (including the first).
        delay: Initial delay in seconds between retries.
        backoff: Factor by which the delay increases each attempt.

    Returns:
        Callable: The decorated function with retry logic.
    """
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    # Log the retry attempt
                    logger.warning(f"Retrying {f.__name__} after {type(e).__name__}: {e} - {mdelay:.2f}s delay, {mtries-1} tries left")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
                except Exception as e:
                    # Log unexpected errors during retry attempts but re-raise immediately
                    logger.error(f"Unexpected error during retry attempt for {f.__name__}: {type(e).__name__} - {e}", exc_info=True)
                    raise # Re-raise unexpected errors immediately
            # Final attempt outside the loop after retries are exhausted
            logger.info(f"Final attempt for {f.__name__}.")
            return f(*args, **kwargs)
        return f_retry
    return deco_retry


class ContradictionEngine:
    """
    The core Contradiction Engine responsible for processing events, detecting
    contradictions based on a defined ruleset, verifying event receipts,
    interacting with a ledger, quarantining failed events, and emitting
    detected contradictions.

    It integrates with a ledger for immutable record-keeping, a tracer for
    auditing, and uses cryptographic signing for integrity. It also includes
    robust error handling, logging, and retry mechanisms.

    Attributes:
        name (str): The name of this engine instance.
        ruleset (List[Callable]): A list of functions (rules) used to detect contradictions.
        ledger (ILedger): An object implementing the ILedger interface for event logging.
        nonce_registry (NonceRegistry): Registry for tracking nonces to prevent replay attacks.
        revocation_registry (RevocationRegistry): Registry for checking revoked certificates.
        verify_strict (bool): Flag to indicate strict receipt verification.
        metabolize_fn (Optional[Callable]): An optional function to process emitted contradictions.
        signing_key (SigningKey): The NaCl signing key used by the engine.
        verify_key (str): The hex-encoded verification key corresponding to the signing key.
        _quarantine_path (Path): Path to the file used for quarantining failed events.
        tracer (Optional[Tracer]): An optional Tracer instance for auditing.
        _running (bool): Flag to control the main processing loop.
        _lock (threading.Lock): A lock for synchronizing access to shared resources (e.g., batch processing).
        _monitor (Optional[ResourceMonitor]): An optional ResourceMonitor instance.
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
        """
        Initializes the ContradictionEngine.

        Args:
            ledger (ILedger): An instance of the ledger interface.
            ruleset (Optional[List[Callable]]): A list of callable functions (rules). Each rule
                                                 should accept an event dictionary and return
                                                 a contradiction dictionary or None.
            signing_key_hex (str): The engine's signing key in hexadecimal format.
            nonce_registry (NonceRegistry): An instance of the nonce registry.
            revocation_registry (RevocationRegistry): An instance of the revocation registry.
            name (str): A unique name for this engine instance (default: "contradiction_engine").
            verify_strict (bool): Whether to perform strict receipt verification (default: True).
            quarantine_path (str): The path for the quarantine log file (default: "data/quarantine.jsonl").
                                   This path is relative to the sandbox root.
            metabolize_fn (Optional[Callable]): An optional function to call after emitting a contradiction.
                                                 It should accept a contradiction dictionary.

        Raises:
            ContradictionEngineError: If required initialization parameters are missing or
                                      if key initialization or sandboxed quarantine setup fails.
        """
        logger.info(f"Initializing ContradictionEngine '{name}'.")
        # Validate required parameters
        if not all([ledger, signing_key_hex, nonce_registry, revocation_registry]):
            logger.error("Missing required initialization parameters.")
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
        self._lock = threading.Lock() # Lock for thread-safe operations, e.g., batch processing

        # Initialize signing and verification keys using nacl
        try:
            # Convert hex string to bytes for nacl SigningKey
            self.signing_key = SigningKey(bytes.fromhex(signing_key_hex))
            # Derive the verify key and encode it to hex for storage/sharing
            self.verify_key = self.signing_key.verify_key.encode(encoder=HexEncoder).decode()
            logger.info("Successfully initialized signing and verification keys.")
        except Exception as e:
            logger.critical(f"Failed to initialize signing keys: {e}", exc_info=True)
            # Raise a specific engine error if key initialization fails
            raise ContradictionEngineError("Failed to initialize signing keys.") from e

        # Set up sandboxed quarantine path
        sandbox_root = Path("data/sandbox")
        try:
            # Ensure the sandbox root directory exists before creating the quarantine path
            sandbox_root.mkdir(parents=True, exist_ok=True)
            # Use the ensure_in_sandbox utility to get the final quarantine path
            self._quarantine_path = ensure_in_sandbox(Path(quarantine_path), sandbox_root)
            logger.info(f"Quarantine path set to: {self._quarantine_path}")
        except Exception as e:
             logger.critical(f"Failed to set up sandboxed quarantine path: {e}", exc_info=True)
             # Raise a specific engine error if sandbox setup fails
             raise ContradictionEngineError("Failed to set up sandboxed quarantine path.") from e


        # Initialize runtime tracer (asynchronous, non-blocking)
        # Tracer initialization is wrapped in try-except as it's not a critical dependency
        self.tracer = None # Initialize to None
        try:
            # Check if Tracer class exists and is a class before attempting to initialize
            if 'Tracer' in globals() and isinstance(Tracer, type):
                self.tracer = Tracer(
                    ledger=self.ledger,
                    # Pass the original hex string if required by Tracer init
                    private_key_hex=signing_key_hex,
                    executor_id=self.name
                )
                logger.info("Tracer initialized successfully.")
            else:
                 logger.warning("Tracer class not found or not a class. Skipping tracer initialization.")

        except Exception as e:
             logger.error(f"Failed to initialize tracer: {e}", exc_info=True)
             # Engine can continue without tracer, so don't raise a critical error here

        # Initialize Resource Monitor if available
        self._running = True # Flag to control the main processing loop
        if 'ResourceMonitor' in globals() and isinstance(ResourceMonitor, type):
            self._monitor = ResourceMonitor("ContradictionEngine")
            logger.info("ResourceMonitor initialized successfully.")
        else:
            self._monitor = None
            logger.warning("ResourceMonitor class not found or not a class. Skipping monitor initialization.")


        logger.info(f"ContradictionEngine '{self.name}' initialization complete.")


    # --------------------------------------------------------
    # Core Methods (Internal)
    # These methods implement the core logic of the engine pipeline.
    # --------------------------------------------------------

    def _sign_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Signs a given payload dictionary using the engine's signing key.

        The payload is first serialized to a JSON string in a canonical (sorted)
        format to ensure consistent signing. The signature and the engine's
        verification key are added to the payload dictionary.

        Args:
            payload (Dict[str, Any]): The dictionary payload to sign. This dictionary
                                       will be modified in place to add signature details.

        Returns:
            Dict[str, Any]: The original payload dictionary with 'signature' (hex-encoded)
                            and 'verify_key' (hex-encoded) fields added.

        Raises:
            ContradictionEngineError: If serialization or signing fails due to underlying
                                      issues (e.g., with nacl or json).
        """
        try:
            # Serialize the payload to a sorted JSON string for consistent signing
            serialized = json.dumps(payload, sort_keys=True).encode('utf-8')
            # Sign the serialized payload using the nacl signing key
            signed_message = self.signing_key.sign(serialized)
            # Extract the hex-encoded signature and add it to the payload
            payload["signature"] = signed_message.signature.hex()
            # Add the engine's hex-encoded verification key to the payload
            payload["verify_key"] = self.verify_key
            logger.debug("Payload signed successfully.")
            return payload
        except Exception as e:
            # Log the error and wrap it in a specific engine exception
            logger.error(f"Failed to sign payload: {e}", exc_info=True)
            raise ContradictionEngineError("Failed to sign payload.") from e


    # Apply Tracer.trace decorator conditionally based on Tracer availability
    if 'Tracer' in globals() and isinstance(Tracer, type) and hasattr(Tracer, 'trace'):
        @Tracer.trace
        def _verify_event(self, event: Dict[str, Any]) -> bool:
            """
            Verifies the validity of an event's receipt.

            This involves calling an external `verify_receipt` function which is
            expected to perform checks like signature validation, nonce uniqueness
            (via NonceRegistry), and revocation status (via RevocationRegistry).

            Args:
                event (Dict[str, Any]): The event dictionary expected to contain a 'receipt' field.

            Returns:
                bool: True if the receipt is valid according to `verify_receipt`, False otherwise.

            Raises:
                VerificationError: If an error occurs during the verification process itself
                                   (e.g., the `verify_receipt` function raises an exception,
                                   or the function is not available).
            """
            receipt = event.get("receipt")
            # Get event ID for logging, default to 'N/A' if not present
            event_id = event.get('id', 'N/A')
            logger.debug(f"Verifying receipt for event: {event_id}")

            # Check for missing receipt. A missing receipt is often a failure condition.
            if not receipt:
                logger.warning(f"Event {event_id} missing receipt.")
                # Depending on the system's policy, missing receipt might be a critical
                # VerificationError, or simply result in the event not being processed
                # further in the pipeline. Returning False here marks it as unverified.
                # raise VerificationError(f"Event {event_id} missing receipt.")
                return False # Return False if receipt is missing but don't raise a critical error

            try:
                # Check if the external verify_receipt function exists and is callable
                if 'verify_receipt' not in globals() or not callable(verify_receipt):
                    logger.error("verify_receipt function not found or not callable.")
                    # This is a configuration/setup error, raise a VerificationError
                    raise VerificationError("verify_receipt function not available.")

                # Call the external receipt verification function
                is_valid = verify_receipt(receipt, strict=self.verify_strict)
                if not is_valid:
                    logger.warning(f"Receipt verification failed for event: {event_id}")
                else:
                     logger.debug(f"Receipt verification succeeded for event: {event_id}")
                return is_valid
            except Exception as e: # Catch potential errors *within* the verify_receipt function
                logger.error(f"Error during receipt verification for event {event_id}: {e}", exc_info=True)
                # Wrap the exception in a VerificationError to indicate a problem
                # with the verification process itself, not just a failed verification result.
                raise VerificationError(f"Receipt verification failed for event {event_id}: {e}") from e
    else:
        # Define a non-traced version if Tracer is not available
        def _verify_event(self, event: Dict[str, Any]) -> bool:
            """
            Verifies the validity of an event's receipt (non-traced version).

            Args:
                event (Dict[str, Any]): The event dictionary.

            Returns:
                bool: True if the receipt is valid, False otherwise.

            Raises:
                VerificationError: If an error occurs during the verification process.
            """
            receipt = event.get("receipt")
            event_id = event.get('id', 'N/A')
            logger.debug(f"Verifying receipt for event: {event_id} (non-traced)")
            if not receipt:
                logger.warning(f"Event {event_id} missing receipt.")
                return False

            try:
                 # Check if verify_receipt function exists and is callable
                if 'verify_receipt' not in globals() or not callable(verify_receipt):
                    logger.error("verify_receipt function not found or not callable.")
                    raise VerificationError("verify_receipt function not available.")

                is_valid = verify_receipt(receipt, strict=self.verify_strict)
                if not is_valid:
                    logger.warning(f"Receipt verification failed for event: {event_id}")
                else:
                     logger.debug(f"Receipt verification succeeded for event: {event_id}")
                return is_valid
            except Exception as e:
                logger.error(f"Error during receipt verification for event {event_id}: {e}", exc_info=True)
                raise VerificationError(f"Receipt verification failed for event {event_id}: {e}") from e


    # Apply Tracer.trace decorator conditionally based on Tracer availability
    if 'Tracer' in globals() and isinstance(Tracer, type) and hasattr(Tracer, 'trace'):
        @Tracer.trace
        def _detect(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
            """
            Runs all configured rules against the event payload to detect contradictions.

            Iterates through the `ruleset` and executes each rule function with
            the event dictionary. Rules that return a non-None dictionary are
            considered to have detected a contradiction. Rule execution errors
            are caught and logged, but do not stop the processing of other rules.

            Args:
                event (Dict[str, Any]): The event dictionary to analyze for contradictions.

            Returns:
                List[Dict[str, Any]]: A list of contradiction dictionaries detected by the rules.
                                     Returns an empty list if no contradictions are found or if
                                     rule execution errors occur (errors are logged).
            """
            contradictions = []
            event_id = event.get('id', 'N/A')
            logger.debug(f"Running contradiction detection rules for event: {event_id}")
            # Iterate through each configured rule in the ruleset
            for rule in self.ruleset:
                # Get rule name for logging, default to 'anonymous_rule' if __name__ is not available
                rule_name = rule.__name__ if hasattr(rule, '__name__') else 'anonymous_rule'
                try:
                    logger.debug(f"Executing rule '{rule_name}' for event: {event_id}")
                    # Execute the rule function with the event
                    result = rule(event)
                    # If the rule returns a non-None result, consider it a detected contradiction
                    if result:
                        logger.info(f"Rule '{rule_name}' detected contradiction for event {event_id}.")
                        contradictions.append(result)
                except Exception as e:
                    # Catch any exception during rule execution. Log the error but continue
                    # processing other rules to maintain robustness.
                    logger.error(f"Rule '{rule_name}' execution failed for event {event_id}: {e}", exc_info=True)
                    # If a single rule failure should stop processing, you would raise
                    # RuleExecutionError here instead of just logging.
            logger.debug(f"Finished running contradiction detection rules for event: {event_id}. Found {len(contradictions)}.")
            return contradictions
    else:
        # Define a non-traced version if Tracer is not available
        def _detect(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
            """
            Runs all configured rules against the event payload to detect contradictions (non-traced).

            Args:
                event (Dict[str, Any]): The event dictionary.

            Returns:
                List[Dict[str, Any]]: A list of contradiction dictionaries.
            """
            contradictions = []
            event_id = event.get('id', 'N/A')
            logger.debug(f"Running contradiction detection rules for event: {event_id} (non-traced)")
            for rule in self.ruleset:
                rule_name = rule.__name__ if hasattr(rule, '__name__') else 'anonymous_rule'
                try:
                    logger.debug(f"Executing rule '{rule_name}' for event: {event_id}")
                    result = rule(event)
                    if result:
                        logger.info(f"Rule '{rule_name}' detected contradiction for event {event_id}.")
                        contradictions.append(result)
                except Exception as e:
                    logger.error(f"Rule '{rule_name}' execution failed for event {event_id}: {e}", exc_info=True)
            logger.debug(f"Finished running contradiction detection rules for event: {event_id}. Found {len(contradictions)}.")
            return contradictions


    # Apply Tracer.trace decorator conditionally based on Tracer availability
    if 'Tracer' in globals() and isinstance(Tracer, type) and hasattr(Tracer, 'trace'):
        @Tracer.trace
        def _quarantine(self, event: Dict[str, Any], reason: str) -> None:
            """
            Writes a failed or problematic event to a durable forensic log file.

            This method is called when an event cannot be processed successfully
            (e.g., due to verification failure, detection error, or emission failure).
            It serializes the event along with a timestamp and the reason
            for quarantine and appends it as a JSON line to the quarantine file.

            Args:
                event (Dict[str, Any]): The event dictionary to quarantine. This should be
                                       the original event that caused the issue.
                reason (str): A description of why the event is being quarantined.

            Raises:
                QuarantineViolation: If a critical error occurs during the file writing process
                                     (e.g., permission denied, disk full), indicating the
                                     quarantine mechanism itself is failing.
            """
            event_id = event.get('id', 'N/A')
            logger.warning(f"Quarantining event {event_id} due to: {reason}")
            try:
                # Create a record dictionary containing timestamp, reason, and the event data
                record = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "reason": reason,
                    "event": event, # Store the full original event
                }
                # Ensure the parent directory for the quarantine file exists before writing
                self._quarantine_path.parent.mkdir(parents=True, exist_ok=True)
                # Open the quarantine file in append mode ('a') with UTF-8 encoding
                with open(self._quarantine_path, "a", encoding="utf-8") as f:
                    # Use json.dump to write the record dictionary to the file
                    json.dump(record, f)
                    # Add a newline character to make the file line-delimited JSON (JSONL)
                    f.write("\n")
                logger.info(f"Event {event_id} quarantined successfully to {self._quarantine_path}")
            except Exception as e:
                # If quarantine fails, this is a critical error as we cannot log the failure.
                # Log the error and raise a QuarantineViolation.
                logger.critical(f"Failed to quarantine event {event_id}: {e}", exc_info=True)
                raise QuarantineViolation(f"Failed to quarantine event {event_id}: {e}") from e
    else:
        # Define a non-traced version if Tracer is not available
        def _quarantine(self, event: Dict[str, Any], reason: str) -> None:
            """
            Writes a failed or problematic event to a durable forensic log file (non-traced).

            Args:
                event (Dict[str, Any]): The event dictionary to quarantine.
                reason (str): A description of why the event is being quarantined.

            Raises:
                QuarantineViolation: If an error occurs during the file writing process.
            """
            event_id = event.get('id', 'N/A')
            logger.warning(f"Quarantining event {event_id} due to: {reason} (non-traced)")
            try:
                record = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "reason": reason,
                    "event": event,
                }
                self._quarantine_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._quarantine_path, "a", encoding="utf-8") as f:
                    json.dump(record, f)
                    f.write("\n")
                logger.info(f"Event {event_id} quarantined successfully to {self._quarantine_path}")
            except Exception as e:
                logger.critical(f"Failed to quarantine event {event_id}: {e}", exc_info=True)
                raise QuarantineViolation(f"Failed to quarantine event {event_id}: {e}") from e


    # Apply Tracer.trace decorator conditionally based on Tracer availability
    # Apply retry decorator to handle transient ledger interaction errors
    if 'Tracer' in globals() and isinstance(Tracer, type) and hasattr(Tracer, 'trace'):
        @Tracer.trace
        @retry(exceptions=LedgerInteractionError, tries=5, delay=2, backoff=3)
        def _emit(self, contradiction: Dict[str, Any]) -> None:
            """
            Signs a detected contradiction and adds it to the ledger.

            This is a critical step in the pipeline. The contradiction payload is
            signed using the engine's key, and the resulting signed event is
            added to the ledger via the `ledger.add_event()` method.
            Interactions with the ledger are retried in case of transient
            `LedgerInteractionError` instances using the `@retry` decorator.
            An optional `metabolize_fn` is called after successful emission to
            allow for further processing of the contradiction.

            Args:
                contradiction (Dict[str, Any]): The contradiction dictionary to emit.
                                               This is the output from a rule in `_detect`.

            Raises:
                ContradictionEngineError: If signing of the contradiction payload fails.
                                          This indicates a problem with the engine's
                                          cryptographic capabilities.
                LedgerInteractionError: If adding the contradiction to the ledger fails
                                        after all configured retry attempts, or if the
                                        ledger interface is invalid.
            """
            # Get contradiction ID for logging, default to 'N/A'
            contradiction_id = contradiction.get('id', 'N/A')
            logger.info(f"Attempting to emit contradiction {contradiction_id} to ledger.")

            # Prepare the base payload structure for the ledger event
            base = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "type": "contradiction", # Explicitly mark this event type
                "payload": contradiction, # Include the original contradiction data
            }

            # Sign the contradiction payload using the engine's signing key
            try:
                signed = self._sign_payload(base)
            except ContradictionEngineError as e:
                # If signing fails, it's a critical engine error. Log and re-raise.
                logger.error(f"Failed to sign contradiction {contradiction_id} before emitting: {e}", exc_info=True)
                raise e # Re-raise signing failure

            # Add the signed contradiction event to the ledger.
            # The @retry decorator will handle transient LedgerInteractionErrors here.
            try:
                # Check if ledger object and its add_event method are valid before calling
                if not hasattr(self.ledger, 'add_event') or not callable(self.ledger.add_event):
                    logger.error("Ledger object missing or has uncallable `add_event()` method.")
                    raise LedgerInteractionError("Ledger missing or has uncallable `add_event()` method.")
                self.ledger.add_event(signed)
                logger.info(f"Contradiction {contradiction_id} successfully added to ledger.")
            except LedgerInteractionError:
                # Re-raise LedgerInteractionError so the retry decorator can catch it.
                # If retries are exhausted, this exception will propagate.
                raise
            except Exception as e:
                # Catch any other exceptions during ledger interaction and wrap them
                # in our custom LedgerInteractionError for consistent handling and retries.
                logger.error(f"Ledger interaction failed for contradiction {contradiction_id}: {e}", exc_info=True)
                raise LedgerInteractionError(f"Failed to add contradiction {contradiction_id} to ledger: {e}") from e

            # Execute the optional metabolism function if provided
            if self.metabolize_fn:
                try:
                    logger.debug(f"Executing metabolism function for contradiction {contradiction_id}.")
                    # Call the metabolism function with the original contradiction payload
                    self.metabolize_fn(contradiction)
                    logger.debug(f"Metabolism function executed for contradiction {contradiction_id}.")
                except Exception as e:
                    # Log metabolism failure but do not stop the emission process.
                    # Metabolism is typically a non-critical post-emission step.
                    logger.warning(f"Metabolism function failed for contradiction {contradiction_id}: {e}", exc_info=True)


            # Increment the contradiction counter if the metric is available and is a Counter
            if isinstance(CONTRADICTION_EVENTS_PROCESSED, Counter):
                 CONTRADICTION_EVENTS_PROCESSED.inc()
                 logger.debug("Contradiction processed count incremented.")
            else:
                 logger.warning("Metrics counter 'CONTRADICTION_EVENTS_PROCESSED' is not a valid Counter instance.")
    else:
         # Define a non-traced version if Tracer is not available
         @retry(exceptions=LedgerInteractionError, tries=5, delay=2, backoff=3) # Apply retry to ledger interaction
         def _emit(self, contradiction: Dict[str, Any]) -> None:
            """
            Signs a detected contradiction and adds it to the ledger (non-traced).

            Args:
                contradiction (Dict[str, Any]): The contradiction dictionary to emit.

            Raises:
                ContradictionEngineError: If signing fails.
                LedgerInteractionError: If adding to the ledger fails after retries.
            """
            contradiction_id = contradiction.get('id', 'N/A')
            logger.info(f"Attempting to emit contradiction {contradiction_id} to ledger (non-traced).")
            base = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "type": "contradiction",
                "payload": contradiction,
            }
            try:
                signed = self._sign_payload(base)
            except ContradictionEngineError as e:
                logger.error(f"Failed to sign contradiction {contradiction_id} before emitting: {e}", exc_info=True)
                raise e

            try:
                if not hasattr(self.ledger, 'add_event') or not callable(self.ledger.add_event):
                    logger.error("Ledger object missing or has uncallable `add_event()` method.")
                    raise LedgerInteractionError("Ledger missing or has uncallable `add_event()` method.")
                self.ledger.add_event(signed)
                logger.info(f"Contradiction {contradiction_id} successfully added to ledger.")
            except LedgerInteractionError:
                raise
            except Exception as e:
                logger.error(f"Ledger interaction failed for contradiction {contradiction_id}: {e}", exc_info=True)
                raise LedgerInteractionError(f"Failed to add contradiction {contradiction_id} to ledger: {e}") from e

            if self.metabolize_fn:
                try:
                    logger.debug(f"Executing metabolism function for contradiction {contradiction_id} (non-traced).")
                    self.metabolize_fn(contradiction)
                    logger.debug(f"Metabolism function executed for contradiction {contradiction_id}.")
                except Exception as e:
                    logger.warning(f"Metabolism function failed for contradiction {contradiction_id}: {e}", exc_info=True)

            if isinstance(CONTRADICTION_EVENTS_PROCESSED, Counter):
                 CONTRADICTION_EVENTS_PROCESSED.inc()
                 logger.debug("Contradiction processed count incremented.")
            else:
                 logger.warning("Metrics counter 'CONTRADICTION_EVENTS_PROCESSED' is not a valid Counter instance.")


    # --------------------------------------------------------
    # Batch / Loop Methods
    # These methods manage the flow of events through the core pipeline.
    # --------------------------------------------------------

    def _run_once_unlocked(self, event: Dict[str, Any]) -> None:
        """
        Processes a single event through the full contradiction detection pipeline.

        This method orchestrates the steps: receipt verification, rule execution
        for contradiction detection, and emission of any detected contradictions.
        It includes error handling at each stage and quarantines events that fail
        verification or lead to critical errors during processing or emission.
        This method is designed to be called *without* acquiring the engine's
        internal lock, as the lock should be managed by the caller (`run_batch`).

        Args:
            event (Dict[str, Any]): The event dictionary to process.
        """
        # Get event ID for logging, default to 'N/A'
        event_id = event.get('id', 'N/A')
        logger.info(f"Processing event: {event_id}")

        # --- Step 1: Verify Event Receipt ---
        try:
            # Call the internal verification method. It can raise VerificationError
            # for process issues or return False for failed verification.
            if not self._verify_event(event):
                # If verification returns False (e.g., invalid signature, invalid nonce)
                # quarantine the original event with a specific reason.
                self._quarantine(event, "Receipt verification failed (returned False)")
                logger.warning(f"Event {event_id} quarantined due to verification failure.")
                return # Stop processing this event further in the pipeline

        except VerificationError as e:
             # If _verify_event raises a VerificationError (problem with the verification process itself)
             # log the critical error and quarantine the original event.
             logger.error(f"Critical verification error for event {event_id}: {e}", exc_info=True)
             self._quarantine(event, f"Critical verification error: {e}")
             return # Stop processing this event

        except Exception as e:
             # Catch any other unexpected errors that might occur during the verification step
             # Log the critical error and quarantine the original event.
             logger.critical(f"Unexpected error during verification for event {event_id}: {e}", exc_info=True)
             self._quarantine(event, f"Unexpected verification error: {e}")
             return # Stop processing this event

        # --- Step 2: Detect Contradictions ---
        try:
            # Run contradiction detection rules against the event
            contradictions = self._detect(event)
            if contradictions:
                logger.info(f"Detected {len(contradictions)} contradictions for event {event_id}.")
            else:
                 logger.info(f"No contradictions detected for event {event_id}.")
        except Exception as e: # Catch any error during the _detect process (including potential RuleExecutionError from rules)
             # If detection fails, log the error and decide whether to quarantine the original event.
             # Here, we choose to quarantine the original event as we couldn't complete detection.
             logger.error(f"Error during contradiction detection for event {event_id}: {e}", exc_info=True)
             self._quarantine(event, f"Error during detection: {e}")
             return # Stop processing this event

        # --- Step 3: Emit Detected Contradictions ---
        # Iterate through each detected contradiction and attempt to emit it
        for c in contradictions:
            try:
                self._emit(c) # _emit includes signing, ledger interaction, metabolism, and retries
            except (LedgerInteractionError, ContradictionEngineError) as e:
                # These are specific errors we handle:
                # - LedgerInteractionError: Emission to ledger failed after all retries.
                # - ContradictionEngineError: Signing the contradiction failed.
                # In these cases, log the error and quarantine the *original event*
                # that led to this contradiction being detected.
                logger.error(f"Failed to emit contradiction for event {event_id} after retries or signing error: {e}", exc_info=True)
                self._quarantine(event, f"Failed to emit contradiction ({c.get('id', 'N/A')}): {e}")
            except Exception as e:
                 # Catch any other unexpected errors that might occur during the emission process
                 # Log the critical error and quarantine the original event.
                 logger.critical(f"Unexpected error during emission for event {event_id}: {e}", exc_info=True)
                 self._quarantine(event, f"Unexpected emission error: {e}")

        logger.info(f"Finished processing event: {event_id}")


    def run_batch(self, events: List[Dict[str, Any]]) -> None:
        """
        Processes a batch of events sequentially.

        Acquires an internal lock to ensure that only one batch is processed
        at a time (useful if `run_batch` is called from multiple threads or
        contexts). Each event in the batch is processed using `_run_once_unlocked`.
        Processing of the current batch and the engine overall stops if a critical
        `QuarantineViolation` occurs (indicating a failure in the quarantine mechanism itself).

        Args:
            events (List[Dict[str, Any]]): A list of event dictionaries to process.

        Raises:
            QuarantineViolation: If a critical error occurs during quarantine that
                                 prevents further processing (e.g., disk full, permissions).
                                 This exception will stop the current batch and the engine.
        """
        logger.info(f"Processing batch of {len(events)} events.")
        # Acquire the lock to ensure thread-safe batch processing
        with self._lock:
            # Iterate through each event in the provided batch
            for i, ev in enumerate(events):
                # Assign a fallback ID for logging if the event dictionary doesn't have one
                event_id = ev.get('id', f'batch_idx_{i}')
                try:
                    logger.debug(f"Processing event {event_id} in batch.")
                    # Process the single event using the unlocked method
                    self._run_once_unlocked(ev)
                except QuarantineViolation as e:
                    # This is a critical failure - the quarantine mechanism itself failed.
                    # Log the critical error, print a message, and stop the engine.
                    logger.critical(f"Quarantine write failure processing event {event_id}: {e}. Stopping engine.", exc_info=True)
                    print(f"[CRITICAL] Quarantine write failure: {e}")
                    self.stop() # Signal the main engine loop to stop
                    break # Stop processing the current batch immediately
                except Exception as e:
                     # Catch any other unexpected errors that might occur during the
                     # processing of a single event within the batch loop.
                     # Log the critical error and continue to the next event in the batch.
                     logger.critical(f"Unexpected error processing event {event_id} in batch: {e}", exc_info=True)


    def run_forever(self, event_source: Callable[[], Optional[Dict[str, Any]]], delay: float = 1.0):
        """
        Continuously polls an event source and processes events in real time.

        This method runs in a loop indefinitely until the engine's internal
        `_running` flag is set to False (e.g., by calling `stop()` or on
        `KeyboardInterrupt`). It calls the provided `event_source` function
        periodically to get new events and processes them using `run_batch`
        (typically with a batch size of one). Includes graceful shutdown on
        `KeyboardInterrupt`.

        Args:
            event_source (Callable[[], Optional[Dict[str, Any]]]): A callable
                function that, when called, returns a single event dictionary
                if one is available, or `None` if no event is currently ready
                for processing.
            delay (float): The time in seconds to wait between polling the event source
                           when no event was received. Defaults to 1.0 seconds.
        """
        logger.info(f"ContradictionEngine '{self.name}' running in forever mode.")
        print(f"[INFO] {self.name} running...")
        # Main loop runs as long as _running flag is True
        while self._running:
            try:
                # Get an event from the source. event_source should be non-blocking or poll.
                event = event_source()
                if event:
                    # Process the received event as a batch of one.
                    # logger.debug("Received event from source.") # Potentially noisy if event source is polled frequently
                    self.run_batch([event])
                else:
                    # If no event is received, just sleep for the specified delay before polling again.
                    # logger.debug("No event from source, sleeping.") # Potentially noisy
                    pass # Just sleep if no event

                # Wait before polling the source again, regardless of whether an event was processed.
                time.sleep(delay)

            except KeyboardInterrupt:
                # Handle graceful shutdown on receiving a KeyboardInterrupt signal (e.g., Ctrl+C)
                logger.info("Keyboard interrupt received. Stopping engine.")
                self.stop() # Call the stop method to set _running to False
            except Exception as e:
                # Catch any unexpected exceptions that might occur in the main loop itself.
                # Log the error and continue running (unless self._running was set to False).
                logger.error(f"Runtime loop exception: {e}", exc_info=True)
                # Decide if runtime loop exception should be quarantined or just logged.
                # Currently logs and continues, but could add quarantine here if needed for forensic analysis.
                # self._quarantine({"error": str(e), "traceback": traceback.format_exc()}, "Runtime loop failure")

        logger.info(f"ContradictionEngine '{self.name}' has stopped.")


    def stop(self):
        """
        Stops the main processing loop and initiates tracer shutdown.

        Sets the internal `_running` flag to False. This signals the `run_forever`
        and `run_batch` loops to exit gracefully after their current iteration
        completes. It also attempts to stop the associated tracer instance if one
        was successfully initialized.
        """
        logger.info(f"ContradictionEngine '{self.name}' is stopping.")
        print(f"[INFO] {self.name} stopping...")
        self._running = False # Signal the main loop(s) to stop
        try:
            # Attempt to stop the tracer gracefully if it exists and has a callable stop method
            if self.tracer and hasattr(self.tracer, 'stop') and callable(self.tracer.stop):
                 self.tracer.stop()
                 logger.info("Tracer stop requested.")
            # If the tracer was running in a separate thread that needs joining, do so here.
            # self.tracer.join(timeout=5) # Example: if tracer._thread existed and was joinable
        except Exception as e:
            # Log errors during tracer shutdown but don't block the engine from stopping
            logger.error(f"Failed to stop tracer cleanly: {e}", exc_info=True)
        logger.info(f"ContradictionEngine '{self.name}' stop process complete.")

    # --------------------------------------------------------
    # Verification / Stats Methods
    # Methods for checking ledger integrity and reporting engine status.
    # --------------------------------------------------------

    def verify_contradiction_chain(self) -> bool:
        """
        Delegates full verification of the ledger's contradiction chain to the ledger itself.

        This method calls the `verify_chain()` method on the configured ledger
        instance to check the integrity and validity of the recorded contradictions.

        Args:
            None

        Returns:
            bool: True if the ledger chain is valid according to the ledger's
                  `verify_chain()` method, False otherwise.

        Raises:
            ContradictionEngineError: If the ledger object does not have a callable
                                      `verify_chain` method, indicating a configuration error.
            LedgerInteractionError: If an error occurs during the ledger chain
                                    verification process itself (e.g., communication
                                    error with the ledger).
        """
        logger.info("Initiating ledger chain verification.")
        # Check if the ledger object has the required verify_chain method and if it's callable
        if not hasattr(self.ledger, "verify_chain") or not callable(self.ledger.verify_chain):
            logger.error("Ledger object missing or has uncallable `verify_chain()` method.")
            # Raise a configuration error if the ledger interface is invalid
            raise ContradictionEngineError("Ledger missing or has uncallable `verify_chain()` method.")
        try:
            # Call the ledger's verification method
            is_valid = self.ledger.verify_chain()
            if is_valid:
                logger.info("Ledger chain verification succeeded.")
            else:
                logger.warning("Ledger chain verification failed.")
            return is_valid
        except Exception as e:
            # Catch any exception from ledger verification and wrap it in a LedgerInteractionError
            logger.error(f"Error during ledger chain verification: {e}", exc_info=True)
            raise LedgerInteractionError(f"Ledger chain verification failed: {e}") from e


    def get_stats(self) -> Dict[str, Any]:
        """
        Reports engine health and operational metrics.

        Gathers statistics on processed contradictions, scars (another potential
        event type in the ledger), quarantine file size, ledger chain validity,
        and engine running status. It attempts to gather as much information as
        possible, logging errors for individual stats components but returning
        partial stats if possible.

        Args:
            None

        Returns:
            Dict[str, Any]: A dictionary containing various engine statistics and status.
                            Includes a 'status' field indicating 'ok' or specific error/warning states
                            encountered while gathering stats.
        """
        logger.info("Gathering engine stats.")
        # Initialize stats dictionary with default values
        stats = {
            "total_contradictions": 0,
            "total_scars": 0,
            "quarantine_size_bytes": 0,
            "chain_valid": False, # Default to False, will be updated by verification attempt
            "engine_running": self._running,
            "status": "ok" # Default status, updated if errors occur during stats gathering
        }
        try:
            # Attempt to get event counts from the ledger if the method is available
            if hasattr(self.ledger, 'get_all_events') and callable(self.ledger.get_all_events):
                try:
                    # Retrieve all events (without verification for performance in stats reporting)
                    all_events = self.ledger.get_all_events(verify=False)
                    # Count contradictions and scars based on the 'type' field
                    stats["total_contradictions"] = len([e for e in all_events if e.get("type") == "contradiction"])
                    stats["total_scars"] = len([e for e in all_events if e.get("type") == "scar"])
                except Exception as e:
                    # Log errors encountered while getting events from the ledger for stats
                    logger.error(f"Failed to get events from ledger for stats: {e}", exc_info=True)
                    # Update the overall status to indicate a ledger stats error
                    stats["status"] = "error_ledger_stats"

            # Attempt to get the quarantine file size if the file exists
            if self._quarantine_path.exists():
                try:
                    stats["quarantine_size_bytes"] = os.path.getsize(self._quarantine_path)
                except Exception as e:
                    # Log errors encountered while getting the quarantine file size
                    logger.warning(f"Failed to get quarantine file size: {e}")
                    stats["quarantine_size_bytes"] = -1 # Indicate error getting size with a negative value
                    # Update the overall status to indicate a warning, but only if it's not already an error
                    if stats["status"] == "ok":
                        stats["status"] = "warning_quarantine_size"

            # Attempt to verify ledger chain validity for the stats report
            try:
                 stats["chain_valid"] = self.verify_contradiction_chain()
            except LedgerInteractionError as e:
                 # Log specific ledger interaction errors during chain verification for stats
                 logger.error(f"Failed to verify chain for stats: {e}", exc_info=True)
                 stats["chain_valid"] = False # Assume invalid if verification process fails
                 # Update status to indicate a chain verification error, if no prior error
                 if stats["status"] == "ok":
                    stats["status"] = "error_chain_verify_stats"
            except Exception as e:
                 # Catch any other unexpected errors during chain verification for stats
                 logger.error(f"Unexpected error verifying chain for stats: {e}", exc_info=True)
                 stats["chain_valid"] = False # Assume invalid if verification process fails
                 # Update status to indicate an unexpected chain verification error, if no prior error
                 if stats["status"] == "ok":
                    stats["status"] = "error_chain_verify_stats_unexpected"


            logger.info(f"Engine stats gathered: {stats}")
            return stats
        except Exception as e:
            # Catch any unexpected error during the overall stats gathering process itself
            logger.error(f"Unexpected error during stats gathering: {e}", exc_info=True)
            stats["status"] = "error_unexpected_stats" # Indicate an overall unexpected stats error
            # Decide whether to raise the exception or return partial stats.
            # Returning partial stats allows monitoring systems to see some information.
            # raise ContradictionEngineError("Failed to gather engine stats.") from e
            return stats # Return partial stats in case of an unexpected error during gathering

# 1. Define placeholder/mock implementations for the external dependencies.
#    These are already defined in the previous cell, so no need to redefine them here.

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
        logger.info(f"Contradiction detected for event {event_id}: Value mismatch.")
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

# 4. Initialize an instance of the ContradictionEngine using the mock dependencies.
#    Need a dummy signing key for demonstration. In production, load securely.
dummy_signing_key_hex = SigningKey.generate().encode(encoder=HexEncoder).decode()
logger.info(f"Using dummy signing key (hex): {dummy_signing_key_hex[:10]}...") # Log snippet

# Instantiate mock dependencies
mock_ledger = ILedger()
mock_nonce_registry = NonceRegistry()
mock_revocation_registry = RevocationRegistry()

# Initialize the engine
engine = ContradictionEngine(
    ledger=mock_ledger,
    ruleset=[example_rule_value_mismatch],
    signing_key_hex=dummy_signing_key_hex,
    nonce_registry=mock_nonce_registry,
    revocation_registry=mock_revocation_registry,
    name="example_engine",
    verify_strict=False, # Set to False for easier demonstration without full receipt structure
    quarantine_path="data/quarantine/example_quarantine.jsonl",
    metabolize_fn=example_metabolize_contradiction
)

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

# Clean up the dummy quarantine file for repeatable runs
if engine._quarantine_path.exists():
    try:
        os.remove(engine._quarantine_path)
        logger.info(f"Cleaned up dummy quarantine file: {engine._quarantine_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up dummy quarantine file: {e}")

# 8. Briefly explain the example code.
print("\nExplanation:")
print("This code demonstrates the basic usage of the ContradictionEngine.")
print("It initializes the engine with mock dependencies, a simple contradiction rule, and a metabolism function.")
print("A batch of example events is created, including some that should trigger the rule and some that shouldn't, and some missing receipts.")
print("The `run_batch` method processes these events.")
print("Events that fail verification (because verify_strict is False and they lack receipts) or trigger the rule are handled.")
print("Events triggering the rule result in a contradiction being emitted (and metabolized by the example function).")
print("Finally, `get_stats` is called to show the engine's state after processing, including contradiction count and quarantine size.")
print("Note: Mock dependencies simulate behavior and errors for demonstration.")
print("In a real application, verify_receipt, ILedger, etc., would be robust implementations.")
print("The signing key used is for demonstration only and must be handled securely in production.")
