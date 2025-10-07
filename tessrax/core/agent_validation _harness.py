"""
agent_validation_harness.py
---------------------------
Tessrax Agent Validation Harness (v3.3 - Standalone Mocks)

Validates agent compliance with protocol rules, semantic tests, and file/ledger integrity.
Includes local mock implementations of dependencies for standalone execution.

Features:
✓ Modular test registry
✓ Chained hash receipts (within harness output)
✓ Optional persistent JSONL logs
✓ Improved readability and test injection
✓ Enhanced error reporting in receipts
✓ **Local Mock Dependencies for Standalone Use**
"""

import json
import hashlib
import time
import uuid
import argparse
import sys # Import sys to check if running in interactive environment
import threading # Import threading for mock Tracer
import traceback # Import traceback for error details
import random # Import random for mock behaviors
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from functools import wraps # Import wraps for decorators


# ---------------------------------------------------------------------
# Local Mock Implementations (for standalone harness execution)
# These replace imports from other Tessrax modules when running this file directly.
# ---------------------------------------------------------------------

import logging # Ensure logging is configured for mocks
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockLedger:
    """Simple mock ledger storing events in memory."""
    def __init__(self, db_path: str = ":memory:"): # Default to in-memory for mock
        logger.info(f"[HARNESS_MOCK] Initialized MockLedger (path: {db_path})")
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock() # Add a lock for thread-safety if needed

    def add_event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            logger.info(f"[HARNESS_MOCK] MockLedger adding event: {event.get('type')}")
            # Simulate adding event with dummy hash/chaining if needed for tests
            event_copy = event.copy()
            event_copy['timestamp'] = time.time() # Add a timestamp
            event_copy['mock_hash'] = hashlib.sha256(json.dumps(event_copy, sort_keys=True).encode()).hexdigest() # Dummy hash
            if self._events:
                 event_copy['mock_prev_hash'] = self._events[-1].get('mock_hash')
            else:
                 event_copy['mock_prev_hash'] = None
            self._events.append(event_copy)

    def verify_chain(self) -> bool:
        logger.info("[HARNESS_MOCK] MockLedger verifying chain (always True)...")
        # Implement simple mock chain verification if needed for tests
        # For now, always return True
        return True

    def get_all_events(self, verify: bool = False) -> List[Dict[str, Any]]:
        logger.info(f"[HARNESS_MOCK] MockLedger getting all events (verify={verify}).")
        # Return copies to prevent external modification
        return [e.copy() for e in self._events]

    def close(self) -> None:
        logger.info("[HARNESS_MOCK] MockLedger closed.")
        self._events = [] # Clear memory on close

class MockNonceRegistry:
    """Simple mock nonce registry."""
    def __init__(self):
        logger.info("[HARNESS_MOCK] Initialized MockNonceRegistry.")
        self._nonces = set()
        self._lock = threading.Lock()

    def check_and_add(self, nonce: str, source: str) -> bool:
        """Checks if nonce exists and adds it if not. Returns True if new/valid, False if duplicate."""
        with self._lock:
            if nonce in self._nonces:
                logger.warning(f"[HARNESS_MOCK] Duplicate nonce detected: {nonce} from {source}")
                return False
            self._nonces.add(nonce)
            logger.debug(f"[HARNESS_MOCK] Registered nonce: {nonce} from {source}")
            return True

class MockRevocationRegistry:
    """Simple mock revocation registry."""
    def __init__(self):
        logger.info("[HARNESS_MOCK] Initialized MockRevocationRegistry.")
        self._revoked_certs = set()
        self._lock = threading.Lock()

    def revoke(self, cert_id: str) -> None:
        """Adds a certificate ID to the revoked list."""
        with self._lock:
            self._revoked_certs.add(cert_id)
            logger.info(f"[HARNESS_MOCK] Revoked certificate: {cert_id}")

    def is_revoked(self, cert_id: str) -> bool:
        """Checks if a certificate ID is in the revoked list."""
        with self._lock:
            is_revoked = cert_id in self._revoked_certs
            logger.debug(f"[HARNESS_MOCK] Checking revocation for {cert_id}: {is_revoked}")
            return is_revoked

class MockTracer:
    """Minimal replacement for the Tessrax Tracer class."""
    def __init__(self, enable_async: bool = True, ledger: Optional[MockLedger] = None, private_key_hex: Optional[str] = None, executor_id: Optional[str] = None):
        logger.info(f"[HARNESS_MOCK] Initialized MockTracer for {executor_id} (async={enable_async}).")
        self.enable_async: bool = enable_async
        self._queue: List[Dict[str, Any]] = []
        self._lock: threading.Lock = threading.Lock()
        self._active: bool = True
        self.ledger: Optional[MockLedger] = ledger # Can link to the mock ledger
        self.private_key_hex: Optional[str] = private_key_hex
        self.executor_id: Optional[str] = executor_id
        # Note: Async thread is not started in this simple dummy

    def record(self, event_type: str, payload: Dict[str, Any]) -> None:
        logger.debug(f"[HARNESS_MOCK_TRACER] Record: {event_type} - {payload}")
        # Optionally add to mock ledger if configured
        if self.ledger:
            try:
                self.ledger.add_event({"type": "TRACE", "payload": {"event_type": event_type, "data": payload}})
            except Exception as e:
                 logger.warning(f"[HARNESS_MOCK_TRACER] Failed to record trace to ledger: {e}")


    def flush(self) -> None:
        logger.debug("[HARNESS_MOCK_TRACER] Flush called (dummy).")

    def stop(self) -> None:
        logger.warning("[HARNESS_MOCK_TRACER] Stop called (dummy).")
        self._active = False # Set active to False

def mock_trace(func: Callable) -> Callable:
    """Simple decorator for tracing function calls (mock)."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
         # logger.debug(f"[HARNESS_MOCK_TRACE_DECORATOR] Calling {func.__name__}") # Uncomment for verbose mock trace
         return func(*args, **kwargs)
    return wrapper

class MockResourceMonitor:
    """Simple mock resource monitor."""
    def __init__(self, name: str = "mock_monitor"):
        logger.info(f"[HARNESS_MOCK] Initialized MockResourceMonitor {name}")

    def start(self) -> None:
        logger.debug("[HARNESS_MOCK_MONITOR] Starting monitor (dummy).")

    def stop(self) -> None:
        logger.debug("[HARNESS_MOCK_MONITOR] Stopping monitor (dummy).")

    def snapshot(self) -> Dict[str, Any]:
        logger.debug("[HARNESS_MOCK_MONITOR] Taking snapshot (dummy).")
        return {"cpu": random.random(), "memory_mb": random.randint(100, 500), "timestamp": time.time()}

    def __enter__(self) -> "MockResourceMonitor":
        self.start()
        return self

    def __exit__(self, exc_type: Optional[type], exc: Optional[Exception], tb: Optional[traceback.TracebackException]) -> None:
        self.stop()
        return False

def mock_ensure_in_sandbox(path: Path, sandbox_root: Path) -> Path:
    """Mocks ensuring a path is within the sandbox root."""
    logger.debug(f"[HARNESS_MOCK_SANDBOX] Ensuring {path} is in sandbox {sandbox_root}")
    # Simulate directory creation without strict sandbox checks
    # In a real scenario, this would validate the path is a child of sandbox_root
    target_path = sandbox_root / path.name # Simplify path handling for mock
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return target_path

def mock_verify_receipt(receipt: Dict[str, Any], strict: bool = True) -> bool:
    """Mocks receipt verification."""
    logger.debug(f"[HARNESS_MOCK_RECEIPT] Verifying receipt (strict={strict})...")
    # Simulate verification success/failure. For the harness demo, let's assume
    # a receipt is valid if it's a dictionary and has a 'dummy_receipt_data' key.
    if not isinstance(receipt, dict) or not receipt:
        logger.warning("[HARNESS_MOCK_RECEIPT] Invalid or empty receipt provided.")
        return False
    # Simple check: does it look like our dummy receipts?
    is_valid = "dummy_receipt_data" in receipt or "receipt_id" in receipt # Accept both dummy types
    logger.debug(f"[HARNESS_MOCK_RECEIPT] Mock verification result: {is_valid}")
    return is_valid


# Alias the mocks to the names expected by the harness code
ILedger = MockLedger
NonceRegistry = MockNonceRegistry
RevocationRegistry = MockRevocationRegistry
Tracer = MockTracer
trace = mock_trace
ResourceMonitor = MockResourceMonitor
ensure_in_sandbox = mock_ensure_in_sandbox
verify_receipt = mock_verify_receipt


# ---------------------------------------------------------------------
# Utility Functions (using local mocks if needed)
# ---------------------------------------------------------------------

def now() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def sha256(data: Any) -> str:
    """Compute SHA-256 hash of canonical JSON representation of data."""
    # Ensure data is serializable; handle non-serializable types gracefully if possible,
    # or rely on standard JSON dumps behavior.
    try:
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    except TypeError as e:
        print(f"[ERROR] Failed to serialize data for hashing: {e}")
        # Return a consistent error hash or raise the exception
        return hashlib.sha256(str(data).encode()).hexdigest() # Fallback to hashing string representation


def append_jsonl(filename: str, data: Dict[str, Any]):
    """Append a dictionary as a JSON line to a file, creating directories if needed."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
    except IOError as e:
        print(f"[ERROR] Failed to append to JSONL file {filename}: {e}")
        # Depending on severity, you might want to raise this exception


# ---------------------------------------------------------------------
# Test Receipt Model
# ---------------------------------------------------------------------

@dataclass
class TestReceipt:
    """Standardized structure for test outcomes."""
    receipt_id: str
    test_id: str
    test_category: str
    target_agent: Optional[str] # Identifier for the agent being tested
    target_artifact: str # Specific component/data being tested (e.g., "Ledger.jsonl", "output", "scar object")
    timestamp: str
    status: str # "PASS", "FAIL", "WARN", "ERROR"
    details: str # Human-readable description of the outcome
    evidence: Dict[str, Any] # Data supporting the outcome (e.g., {"expected": ..., "actual": ...})
    error: Optional[str] = None # Details of any exception caught during the test
    prev_hash: Optional[str] = None # Hash of the previous receipt
    current_hash: Optional[str] = None # Hash of this receipt

    @classmethod
    def create(cls, *, test_id: str, category: str, target_agent: Optional[str], target_artifact: str,
               status: str, details: str, expected: Any, actual: Any, prev_hash: Optional[str],
               error: Optional[str] = None) -> 'TestReceipt':
        """Factory method to create a TestReceipt with auto-generated ID and hash."""
        # Create the base dictionary before computing the hash
        base = {
            "receipt_id": f"TEST-{uuid.uuid4()}",
            "test_id": test_id,
            "test_category": category,
            "target_agent": target_agent,
            "target_artifact": target_artifact,
            "timestamp": now(),
            "status": status,
            "details": details,
            "evidence": {"expected": expected, "actual": actual},
            "error": error,
            "prev_hash": prev_hash # Include prev_hash in the data being hashed
        }
        # Compute the current hash based on the content (including prev_hash)
        current_hash = sha256(base)
        # Add the current_hash to the dictionary
        base["current_hash"] = current_hash
        # Return the dataclass instance
        return cls(**base)


# ---------------------------------------------------------------------
# Test Input Context
# ---------------------------------------------------------------------

@dataclass
class TestContext:
    """
    Contextual data provided to test functions.
    Allows injecting specific states or data for testing.
    """
    ledger_entries: List[Dict[str, Any]]
    file_manifest: Dict[str, str] # Mapping of file paths to expected SHA-256 hashes
    scar_object: Dict[str, Any] # Example SCAR (Systematic Contradiction Analysis Record) object
    output_text: str # Example agent output text
    semantic_engine: Any # Mock or real semantic engine interface


# ---------------------------------------------------------------------
# Test Definitions
# Functions that implement specific validation checks.
# Each test function accepts TestContext and the previous receipt's hash,
# and returns a TestReceipt.
# ---------------------------------------------------------------------

# === Integrity & Provenance Audits (IPA) ===

def test_ledger_chain(ctx: TestContext, prev_hash: Optional[str]) -> TestReceipt:
    """IPA-01: Verifies the hash chain integrity of ledger entries."""
    prev_entry_hash = None
    try:
        if not ctx.ledger_entries:
             # Consider this a PASS if no entries, as the chain isn't broken
             return TestReceipt.create(
                test_id="IPA-01", category="IPA", target_agent=None, target_artifact="Mock Ledger entries",
                status="PASS", details="No ledger entries to check.",
                expected="At least one entry or empty list", actual="Empty list",
                prev_hash=prev_hash
            )

        for i, entry in enumerate(ctx.ledger_entries):
            # Ensure required mock hash fields are present in each entry
            if "mock_hash" not in entry or ("mock_prev_hash" not in entry and i > 0):
                 return TestReceipt.create(
                    test_id="IPA-01", category="IPA", target_agent=None, target_artifact=f"Mock Ledger entry index {i}",
                    status="FAIL", details="Mock Ledger entry missing required hash fields.",
                    expected='"mock_hash" and "mock_prev_hash" (if not first entry)', actual=list(entry.keys()),
                    prev_hash=prev_hash
                )

            # Check the previous hash link using mock hash fields
            if i > 0 and entry.get("mock_prev_hash") != prev_entry_hash:
                return TestReceipt.create(
                    test_id="IPA-01", category="IPA", target_agent=None, target_artifact=f"Mock Ledger entry index {i}",
                    status="FAIL", details="Mock hash chain broken.",
                    expected=prev_entry_hash, actual=entry.get("mock_prev_hash"),
                    prev_hash=prev_hash
                )
            # Update the hash of the current entry for the next iteration's check
            prev_entry_hash = entry.get("mock_hash")

        # If the loop completes without finding a break, the mock chain is intact
        return TestReceipt.create(
            test_id="IPA-01", category="IPA", target_agent=None, target_artifact="Mock Ledger entries",
            status="PASS", details="Mock ledger chain intact.",
            expected="Chain continuity", actual="OK",
            prev_hash=prev_hash
        )
    except Exception as e:
        # Catch any unexpected errors during the test itself
        return TestReceipt.create(
            test_id="IPA-01", category="IPA", target_agent=None, target_artifact="Mock Ledger entries",
            status="ERROR", details="Exception during test execution.",
            expected=None, actual=None, error=str(e),
            prev_hash=prev_hash
        )


def test_file_hashes(ctx: TestContext, prev_hash: Optional[str]) -> TestReceipt:
    """IPA-02: Verifies file integrity against a provided manifest."""
    mismatches = {}
    errors = []
    try:
        if not ctx.file_manifest:
             return TestReceipt.create(
                test_id="IPA-02", category="IPA", target_agent=None, target_artifact="File manifest",
                status="PASS", details="No file manifest provided.",
                expected="Non-empty manifest or empty dict", actual="Empty dict",
                prev_hash=prev_hash
            )

        for file, expected in ctx.file_manifest.items():
            path = Path(file)
            if not path.exists():
                mismatches[file] = "MISSING"
                errors.append(f"File missing: {file}")
            else:
                try:
                    actual = hashlib.sha256(path.read_bytes()).hexdigest()
                    if actual != expected:
                        mismatches[file] = actual
                        errors.append(f"Hash mismatch for {file}: Expected {expected[:8]}..., Got {actual[:8]}...")
                except Exception as e:
                     mismatches[file] = f"ERROR: {e}"
                     errors.append(f"Error reading file {file}: {e}")

        status = "FAIL" if mismatches else "PASS"
        details = "File hash validation completed."
        if errors:
            details += f" Issues: {'; '.join(errors)}"

        return TestReceipt.create(
            test_id="IPA-02", category="IPA", target_agent=None, target_artifact="File manifest",
            status=status, details=details,
            expected=ctx.file_manifest, actual=mismatches or "all match",
            prev_hash=prev_hash
        )
    except Exception as e:
        # Catch any unexpected errors during the test itself
        return TestReceipt.create(
            test_id="IPA-02", category="IPA", target_agent=None, target_artifact="File manifest",
            status="ERROR", details="Exception during test execution.",
            expected=None, actual=None, error=str(e),
            prev_hash=prev_hash
        )


# === Protocol Compliance Gauntlet (PCG) ===

def test_signature_lock(ctx: TestContext, prev_hash: Optional[str]) -> TestReceipt:
    """PCG-01: Checks for expected start and end patterns in agent output."""
    try:
        if not isinstance(ctx.output_text, str):
             return TestReceipt.create(
                test_id="PCG-01", category="PCG", target_agent="agent", target_artifact="output text",
                status="ERROR", details="Agent output is not a string.",
                expected="string", actual=type(ctx.output_text).__name__,
                prev_hash=prev_hash
            )
        output = ctx.output_text.strip()
        start_pattern = "GPT to Josh—"
        end_pattern = "-Tessrax LLC-"

        starts_correctly = output.startswith(start_pattern)
        ends_correctly = output.endswith(end_pattern)

        status = "PASS" if (starts_correctly and ends_correctly) else "FAIL"
        details = "Output signature pattern check."
        if not starts_correctly:
            details += f" Does not start with '{start_pattern}'."
        if not ends_correctly:
            details += f" Does not end with '{end_pattern}'."
        if status == "PASS":
             details = "Output signature patterns match."

        return TestReceipt.create(
            test_id="PCG-01", category="PCG", target_agent="agent", target_artifact="output text",
            status=status, details=details,
            expected=f"Starts with '{start_pattern}' and ends with '{end_pattern}'",
            actual=output[:60] + "..." + output[-60:] if len(output) > 120 else output,
            prev_hash=prev_hash
        )
    except Exception as e:
        # Catch any unexpected errors during the test itself
        return TestReceipt.create(
            test_id="PCG-01", category="PCG", target_agent="agent", target_artifact="output text",
            status="ERROR", details="Exception during test execution.",
            expected=None, actual=None, error=str(e),
            prev_hash=prev_hash
        )


def test_scar_schema(ctx: TestContext, prev_hash: Optional[str]) -> TestReceipt:
    """PCG-02: Validates required fields in a SCAR object."""
    try:
        if not isinstance(ctx.scar_object, dict):
            return TestReceipt.create(
               test_id="PCG-02", category="PCG", target_agent="agent", target_artifact="scar object",
               status="ERROR", details="SCAR object is not a dictionary.",
               expected="dictionary", actual=type(ctx.scar_object).__name__,
               prev_hash=prev_hash
           )

        required_fields = ["scar_id", "status", "impact_score"]
        actual_fields = list(ctx.scar_object.keys())
        missing_fields = [k for k in required_fields if k not in actual_fields]

        status = "PASS" if not missing_fields else "FAIL"
        details = "SCAR schema compliance check."
        if missing_fields:
            details += f" Missing required fields: {', '.join(missing_fields)}."
        if status == "PASS":
             details = "SCAR schema compliant."

        return TestReceipt.create(
            test_id="PCG-02", category="PCG", target_agent="agent", target_artifact="scar object",
            status=status, details=details,
            expected=required_fields, actual=actual_fields,
            prev_hash=prev_hash
        )
    except Exception as e:
        # Catch any unexpected errors during the test itself
        return TestReceipt.create(
            test_id="PCG-02", category="PCG", target_agent="agent", target_artifact="scar object",
            status="ERROR", details="Exception during test execution.",
            expected=None, actual=None, error=str(e),
            prev_hash=prev_hash
        )


# === Semantic Contradiction Drills (SCD) ===

def test_liar_paradox(ctx: TestContext, prev_hash: Optional[str]) -> TestReceipt:
    """SCD-01: Checks if the semantic engine acknowledges the Liar Paradox."""
    expected_keywords = ["contradiction", "paradox", "false statement", "neither true nor false"] # Expanded keywords for better matching
    actual_response = ""
    try:
        if not hasattr(ctx.semantic_engine, 'respond') or not callable(ctx.semantic_engine.respond):
             return TestReceipt.create(
                test_id="SCD-01", category="SCD", target_agent="agent", target_artifact="semantic engine",
                status="WARN", details="Semantic engine mock missing 'respond' method.",
                expected="Callable 'respond' method", actual=str(type(ctx.semantic_engine)),
                prev_hash=prev_hash
            )

        # Use a more standard Liar Paradox phrasing
        question = "Is the statement 'This statement is false' true or false?"
        actual_response = ctx.semantic_engine.respond(question)
        response_lower = actual_response.lower()

        # Check if any of the expected keywords are in the response
        acknowledges_paradox = any(x in response_lower for x in expected_keywords)

        status = "PASS" if acknowledges_paradox else "FAIL"
        details = "Liar paradox check."
        if status == "PASS":
             details = "Semantic engine acknowledged liar paradox."
        else:
             details = "Semantic engine did not clearly acknowledge liar paradox."


        return TestReceipt.create(
            test_id="SCD-01", category="SCD", target_agent="agent", target_artifact="semantic engine",
            status=status, details=details,
            expected=f"Response containing one of: {expected_keywords}", actual=actual_response,
            prev_hash=prev_hash
        )
    except Exception as e:
        # Catch any unexpected errors during the test itself (e.g., semantic_engine.respond raises an error)
        return TestReceipt.create(
            test_id="SCD-01", category="SCD", target_agent="agent", target_artifact="semantic engine",
            status="ERROR", details="Exception during test execution.",
            expected=None, actual=actual_response, error=str(e),
            prev_hash=prev_hash
        )


def test_russell_paradox(ctx: TestContext, prev_hash: Optional[str]) -> TestReceipt:
    """SCD-02: Checks if the semantic engine acknowledges Russell's Paradox."""
    expected_keywords = ["recursive", "contradiction", "undefined", "paradox", "set of all sets", "does not contain itself"] # Expanded keywords
    actual_response = ""
    try:
        if not hasattr(ctx.semantic_engine, 'respond') or not callable(ctx.semantic_engine.respond):
             return TestReceipt.create(
                test_id="SCD-02", category="SCD", target_agent="agent", target_artifact="semantic engine",
                status="WARN", details="Semantic engine mock missing 'respond' method.",
                expected="Callable 'respond' method", actual=str(type(ctx.semantic_engine)),
                prev_hash=prev_hash
            )

        question = "Does the set of all sets that do not contain themselves contain itself?"
        actual_response = ctx.semantic_engine.respond(question)
        response_lower = actual_response.lower()

        # Check if any of the expected keywords are in the response
        acknowledges_paradox = any(x in response_lower for x in expected_keywords)

        status = "PASS" if acknowledges_paradox else "FAIL"
        details = "Russell paradox check."
        if status == "PASS":
             details = "Semantic engine acknowledged Russell's paradox."
        else:
             details = "Semantic engine did not clearly acknowledge Russell's paradox."

        return TestReceipt.create(
            test_id="SCD-02", category="SCD", target_agent="agent", target_artifact="semantic engine",
            status=status, details=details,
            expected=f"Response containing one of: {expected_keywords}", actual=actual_response,
            prev_hash=prev_hash
        )
    except Exception as e:
        # Catch any unexpected errors during the test itself
        return TestReceipt.create(
            test_id="SCD-02", category="SCD", target_agent="agent", target_artifact="semantic engine",
            status="ERROR", details="Exception during test execution.",
            expected=None, actual=actual_response, error=str(e),
            prev_hash=prev_hash
        )


# ---------------------------------------------------------------------
# Test Registry
# List of all test functions to be run.
# ---------------------------------------------------------------------

TEST_SUITE: List[Callable[[TestContext, Optional[str]], TestReceipt]] = [
    test_ledger_chain,
    test_file_hashes,
    test_signature_lock,
    test_scar_schema,
    test_liar_paradox,
    test_russell_paradox
]


# ---------------------------------------------------------------------
# Test Runner
# Orchestrates test execution and receipt logging.
# ---------------------------------------------------------------------

def run_all_tests(ctx: TestContext, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Runs all tests in the TEST_SUITE and optionally logs receipts.
    Receipts are chained using hashes.
    """
    receipts = []
    prev_hash = None

    print(f"Running {len(TEST_SUITE)} validation tests...")

    for test_func in TEST_SUITE:
        test_name = test_func.__name__
        # Extract category from test_id prefix (e.g., "IPA-01" -> "IPA")
        test_id_prefix = test_name.split('_')[0].upper() if '_' in test_name else "UNCATEGORIZED"
        print(f"  Running {test_name} ({test_id_prefix})...")
        try:
            # Execute the test function with the context and previous hash
            # Pass test_id_prefix as category if test function doesn't provide it
            receipt = test_func(ctx, prev_hash)
            # Ensure receipt has a category, use prefix if not set by test function
            if not receipt.test_category:
                 receipt.test_category = test_id_prefix

            print(f"    Result: {receipt.status}")
        except Exception as e:
            # This catches exceptions *within* the test runner itself,
            # not exceptions raised by the agent code being tested (those should
            # ideally be caught and reported by the test functions themselves).
            # This is a safeguard for faulty test functions.
            receipt = TestReceipt.create(
                test_id=f"{test_id_prefix}-HARNESS-EXC", category="HARNESS_ERROR", target_agent=None, target_artifact=test_name,
                status="ERROR", details=f"Exception in harness while running test: {e}",
                expected=None, actual=None, error=str(e), prev_hash=prev_hash
            )
            print(f"    Result: ERROR (Harness Exception)")
            # Log traceback for harness errors
            logger.error(f"Harness exception during test {test_name}:", exc_info=True)


        # Update the previous hash for the next receipt
        prev_hash = receipt.current_hash

        # Log the receipt if an output file is specified
        if output_file:
            try:
                append_jsonl(output_file, asdict(receipt))
            except Exception as e:
                print(f"[ERROR] Failed to write receipt to output file {output_file}: {e}")
                logger.error(f"Failed to write receipt to {output_file}: {e}", exc_info=True)
                # Decide if this failure should stop the harness or just warn

        # Add the receipt (as dict) to the results list
        receipts.append(asdict(receipt))

    print("Validation test run complete.")
    return receipts


# ---------------------------------------------------------------------
# CLI Execution
# Allows running the harness directly from command line or in a notebook.
# ---------------------------------------------------------------------

# Check if the script is being run directly (not imported) and not in an interactive environment like Colab
# This prevents the argparse from running and potentially raising SystemExit in a notebook cell
if __name__ == "__main__" and not hasattr(sys, 'ps1'):
    parser = argparse.ArgumentParser(description="Run Tessrax Agent Validation Harness.")
    parser.add_argument("--out", type=str, help="Optional output .jsonl file for receipts.")
    # Add arguments for specifying mock data paths or configurations if needed later

    args = parser.parse_args()

    # --- Mock context setup ---
    # This section provides example data for the tests.
    # In a real scenario, this would be populated from actual agent outputs,
    # ledger files, file system scans, etc.
    print("Setting up mock test context...")

    # Example Ledger Entries (simulating a simple chain)
    # Note: In a real system, these would be loaded from a ledger file/DB.
    # We manually create a simple chain here for demonstration using mock hash fields.
    mock_ledger_entry_1_payload = {"type": "claim", "agent": "A", "data": "payload1"}
    mock_ledger_entry_1_hash = hashlib.sha256(json.dumps(mock_ledger_entry_1_payload, sort_keys=True).encode()).hexdigest()
    mock_ledger_entry_1 = {"timestamp": now(), "type": "claim", "payload": mock_ledger_entry_1_payload,
                           "mock_prev_hash": None, "mock_hash": mock_ledger_entry_1_hash}

    mock_ledger_entry_2_payload = {"type": "scar", "agent": "B", "data": "payload2"}
    mock_ledger_entry_2_hash = hashlib.sha256(json.dumps(mock_ledger_entry_2_payload, sort_keys=True).encode()).hexdigest()
    mock_ledger_entry_2 = {"timestamp": now(), "type": "scar", "payload": mock_ledger_entry_2_payload,
                           "mock_prev_hash": mock_ledger_entry_1_hash, "mock_hash": mock_ledger_entry_2_hash}

    # Example of a broken chain entry (optional, for testing failure case)
    # mock_ledger_entry_broken_payload = {"type": "scar", "agent": "C", "data": "broken"}
    # mock_ledger_entry_broken = {"timestamp": now(), "type": "scar", "payload": mock_ledger_entry_broken_payload,
    #                            "mock_prev_hash": "wrong_hash_intentionally", # This will break the chain check
    #                            "mock_hash": hashlib.sha256(json.dumps(mock_ledger_entry_broken_payload, sort_keys=True).encode()).hexdigest()}


    # Example File Manifest (mapping expected file hashes)
    # In a real test, you'd compute the expected hashes of known good files.
    # For this mock, we'll create a dummy file and get its hash.
    dummy_file_path = Path("data/test_files/dummy_agent_output.txt")
    dummy_file_content = "This is a dummy file content for testing file integrity."
    # Use the local mock_ensure_in_sandbox to handle the path
    sandboxed_dummy_file_path = mock_ensure_in_sandbox(dummy_file_path, Path("data/sandbox")) # Ensure it's relative to a sandbox root
    sandboxed_dummy_file_path.write_text(dummy_file_content, encoding="utf-8")
    dummy_file_hash = hashlib.sha256(dummy_file_content.encode()).hexdigest()

    mock_file_manifest = {
        str(sandboxed_dummy_file_path): dummy_file_hash,
        # Add another file that might be missing or have wrong content for testing
        # str(mock_ensure_in_sandbox(Path("data/test_files/another_file.log"), Path("data/sandbox"))): "expected_hash_of_another_file", # Example missing/bad file
    }

    # Example SCAR Object (Systematic Contradiction Analysis Record)
    mock_scar_object = {
        "scar_id": "SCAR-XYZ-789",
        "status": "open",
        "impact_score": 75,
        "details": "Mismatch detected in report vs claim.",
        "related_claims": ["claim-123", "report-456"],
        "timestamp": now()
        # Missing 'fuel' field from ContradictionEngine's Scar dataclass - will fail PCG-02 if required there
    }
    # Example SCAR object that would fail the schema check
    # mock_scar_object_bad_schema = {"id": "SCAR-ABC-111", "status": "resolved"} # Missing impact_score


    # Example Agent Output Text
    mock_output_text = "GPT to Josh—Processing complete. Analysis successful.-Tessrax LLC-"
    # Example output that would fail the signature lock test:
    # mock_output_text_bad_sig = "Processing complete. Analysis successful."


    # Example Semantic Engine Mock
    # This mock simulates a semantic engine's 'respond' method.
    # It's designed to provide responses that allow the semantic tests to pass.
    class MockSemanticEngine:
        def respond(self, query: str) -> str:
            query_lower = query.lower()
            if "liar" in query_lower or "this statement is false" in query_lower:
                return "That statement presents a logical contradiction and is neither true nor false." # Response acknowledging paradox
            elif "set of all sets that do not contain themselves" in query_lower or "russell" in query_lower:
                 return "That query leads to Russell's paradox, a fundamental contradiction in naive set theory involving sets that do not contain themselves." # Response acknowledging paradox
            else:
                return f"Acknowledged query: '{query}'."

    mock_semantic_engine_instance = MockSemanticEngine()

    # Create the TestContext instance with the mock data
    ctx = TestContext(
        ledger_entries=[mock_ledger_entry_1, mock_ledger_entry_2], # Add your mock ledger entries here (or include the broken one)
        file_manifest=mock_file_manifest, # Add your mock file manifest here
        scar_object=mock_scar_object, # Add your mock SCAR object here (or the bad schema one)
        output_text=mock_output_text, # Add your mock agent output text here (or the bad sig one)
        semantic_engine=mock_semantic_engine_instance # Add your mock semantic engine here
    )

    # Run the tests using the created context
    results = run_all_tests(ctx, output_file=args.out)

    # Print a summary of the results
    print("\n--- Test Results Summary ---")
    for r in results:
        status = r.get('status', 'UNKNOWN')
        prefix = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️" if status == "WARN" else "❗"
        test_id = r.get('test_id', 'N/A')
        category = r.get('test_category', 'N/A')
        details = r.get('details', 'No details provided')
        print(f"{prefix} {test_id} ({category}): {details}")

    # Clean up dummy file and directory
    if sandboxed_dummy_file_path.exists():
        try:
            sandboxed_dummy_file_path.unlink()
            # Attempt to remove the directory only if it's empty
            try:
                 sandboxed_dummy_file_path.parent.rmdir()
                 logger.info(f"Cleaned up dummy directory: {sandboxed_dummy_file_path.parent}")
            except OSError:
                 logger.debug(f"Directory not empty, skipping rmdir: {sandboxed_dummy_file_path.parent}")
            logger.info(f"Cleaned up dummy file: {sandboxed_dummy_file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up dummy file/directory: {e}")
