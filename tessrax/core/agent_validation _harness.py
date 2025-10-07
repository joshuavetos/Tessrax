"""
agent_validation_harness.py
---------------------------
Tessrax Agent Validation Harness (v3.2 Refactor)

Validates agent compliance with protocol rules, semantic tests, and file/ledger integrity.

Features:
✓ Modular test registry
✓ Chained hash receipts (within harness output)
✓ Optional persistent JSONL logs
✓ Improved readability and test injection
✓ Enhanced error reporting in receipts
"""

import json
import hashlib
import time
import uuid
import argparse
import sys # Import sys to check if running in interactive environment
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------
# Utility Functions
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
                test_id="IPA-01", category="IPA", target_agent=None, artifact="Ledger entries",
                status="PASS", details="No ledger entries to check.",
                expected="At least one entry or empty list", actual="Empty list",
                prev_hash=prev_hash
            )

        for i, entry in enumerate(ctx.ledger_entries):
            # Ensure required fields are present in each entry
            if "current_hash" not in entry or ("prev_hash" not in entry and i > 0):
                 return TestReceipt.create(
                    test_id="IPA-01", category="IPA", target_agent=None, artifact=f"Ledger entry index {i}",
                    status="FAIL", details="Ledger entry missing required hash fields.",
                    expected='"current_hash" and "prev_hash" (if not first entry)', actual=list(entry.keys()),
                    prev_hash=prev_hash
                )

            # Check the previous hash link
            if i > 0 and entry.get("prev_hash") != prev_entry_hash:
                return TestReceipt.create(
                    test_id="IPA-01", category="IPA", target_agent=None, artifact=f"Ledger entry index {i}",
                    status="FAIL", details="Hash chain broken.",
                    expected=prev_entry_hash, actual=entry.get("prev_hash"),
                    prev_hash=prev_hash
                )
            # Update the hash of the current entry for the next iteration's check
            prev_entry_hash = entry.get("current_hash")

        # If the loop completes without finding a break, the chain is intact
        return TestReceipt.create(
            test_id="IPA-01", category="IPA", target_agent=None, artifact="Ledger entries",
            status="PASS", details="Ledger chain intact.",
            expected="Chain continuity", actual="OK",
            prev_hash=prev_hash
        )
    except Exception as e:
        # Catch any unexpected errors during the test itself
        return TestReceipt.create(
            test_id="IPA-01", category="IPA", target_agent=None, artifact="Ledger entries",
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
                test_id="IPA-02", category="IPA", target_agent=None, artifact="File manifest",
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
            test_id="IPA-02", category="IPA", target_agent=None, artifact="File manifest",
            status=status, details=details,
            expected=ctx.file_manifest, actual=mismatches or "all match",
            prev_hash=prev_hash
        )
    except Exception as e:
        # Catch any unexpected errors during the test itself
        return TestReceipt.create(
            test_id="IPA-02", category="IPA", target_agent=None, artifact="File manifest",
            status="ERROR", details="Exception during test execution.",
            expected=None, actual=None, error=str(e),
            prev_hash=prev_hash
        )


# === Protocol Compliance Gauntlet (PCG) ===

def test_signature_lock(ctx: TestContext, prev_hash: Optional[str]) -> TestReceipt:
    """PCG-01: Checks for expected start and end patterns in agent output."""
    try:
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
            test_id="PCG-01", category="PCG", target_agent="agent", artifact="output text",
            status=status, details=details,
            expected=f"Starts with '{start_pattern}' and ends with '{end_pattern}'",
            actual=output[:60] + "..." + output[-60:] if len(output) > 120 else output,
            prev_hash=prev_hash
        )
    except Exception as e:
        # Catch any unexpected errors during the test itself (e.g., output_text is not a string)
        return TestReceipt.create(
            test_id="PCG-01", category="PCG", target_agent="agent", artifact="output text",
            status="ERROR", details="Exception during test execution.",
            expected=None, actual=None, error=str(e),
            prev_hash=prev_hash
        )


def test_scar_schema(ctx: TestContext, prev_hash: Optional[str]) -> TestReceipt:
    """PCG-02: Validates required fields in a SCAR object."""
    try:
        required_fields = ["scar_id", "status", "impact_score"]
        actual_fields = list(ctx.scar_object.keys()) if isinstance(ctx.scar_object, dict) else []
        missing_fields = [k for k in required_fields if k not in actual_fields]

        status = "PASS" if not missing_fields else "FAIL"
        details = "SCAR schema compliance check."
        if missing_fields:
            details += f" Missing required fields: {', '.join(missing_fields)}."
        if status == "PASS":
             details = "SCAR schema compliant."

        return TestReceipt.create(
            test_id="PCG-02", category="PCG", target_agent="agent", artifact="scar object",
            status=status, details=details,
            expected=required_fields, actual=actual_fields,
            prev_hash=prev_hash
        )
    except Exception as e:
        # Catch any unexpected errors during the test itself (e.g., scar_object is not a dict)
        return TestReceipt.create(
            test_id="PCG-02", category="PCG", target_agent="agent", artifact="scar object",
            status="ERROR", details="Exception during test execution.",
            expected=None, actual=None, error=str(e),
            prev_hash=prev_hash
        )


# === Semantic Contradiction Drills (SCD) ===

def test_liar_paradox(ctx: TestContext, prev_hash: Optional[str]) -> TestReceipt:
    """SCD-01: Checks if the semantic engine acknowledges the Liar Paradox."""
    expected_keywords = ["contradiction", "paradox", "false statement"] # Expanded keywords
    actual_response = ""
    try:
        if not hasattr(ctx.semantic_engine, 'respond') or not callable(ctx.semantic_engine.respond):
             return TestReceipt.create(
                test_id="SCD-01", category="SCD", target_agent="agent", artifact="semantic engine",
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
            test_id="SCD-01", category="SCD", target_agent="agent", artifact="semantic engine",
            status=status, details=details,
            expected=f"Response containing one of: {expected_keywords}", actual=actual_response,
            prev_hash=prev_hash
        )
    except Exception as e:
        # Catch any unexpected errors during the test itself (e.g., semantic_engine.respond raises an error)
        return TestReceipt.create(
            test_id="SCD-01", category="SCD", target_agent="agent", artifact="semantic engine",
            status="ERROR", details="Exception during test execution.",
            expected=None, actual=actual_response, error=str(e),
            prev_hash=prev_hash
        )


def test_russell_paradox(ctx: TestContext, prev_hash: Optional[str]) -> TestReceipt:
    """SCD-02: Checks if the semantic engine acknowledges Russell's Paradox."""
    expected_keywords = ["recursive", "contradiction", "undefined", "paradox", "set of all sets"] # Expanded keywords
    actual_response = ""
    try:
        if not hasattr(ctx.semantic_engine, 'respond') or not callable(ctx.semantic_engine.respond):
             return TestReceipt.create(
                test_id="SCD-02", category="SCD", target_agent="agent", artifact="semantic engine",
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
            test_id="SCD-02", category="SCD", target_agent="agent", artifact="semantic engine",
            status=status, details=details,
            expected=f"Response containing one of: {expected_keywords}", actual=actual_response,
            prev_hash=prev_hash
        )
    except Exception as e:
        # Catch any unexpected errors during the test itself
        return TestReceipt.create(
            test_id="SCD-02", category="SCD", target_agent="agent", artifact="semantic engine",
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
        print(f"  Running {test_name}...")
        try:
            # Execute the test function with the context and previous hash
            receipt = test_func(ctx, prev_hash)
            print(f"    Result: {receipt.status}")
        except Exception as e:
            # This catches exceptions *within* the test runner itself,
            # not exceptions raised by the agent code being tested (those should
            # ideally be caught and reported by the test functions themselves).
            # This is a safeguard for faulty test functions.
            receipt = TestReceipt.create(
                test_id="HARNESS-EXC", category="HARNESS_ERROR", target_agent=None, artifact=test_name,
                status="ERROR", details=f"Exception in harness while running test: {e}",
                expected=None, actual=None, error=str(e), prev_hash=prev_hash
            )
            print(f"    Result: ERROR (Harness Exception)")

        # Update the previous hash for the next receipt
        prev_hash = receipt.current_hash

        # Log the receipt if an output file is specified
        if output_file:
            try:
                append_jsonl(output_file, asdict(receipt))
            except Exception as e:
                print(f"[ERROR] Failed to write receipt to output file {output_file}: {e}")
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
    # We manually create a simple chain here for demonstration.
    mock_ledger_entry_1_payload = {"type": "claim", "agent": "A", "data": "payload1"}
    mock_ledger_entry_1_hash = sha256(mock_ledger_entry_1_payload) # Hash of payload + None prev_hash
    mock_ledger_entry_1 = {"timestamp": now(), "type": "claim", "payload": mock_ledger_entry_1_payload,
                           "prev_hash": None, "current_hash": mock_ledger_entry_1_hash}

    mock_ledger_entry_2_payload = {"type": "scar", "agent": "B", "data": "payload2"}
    mock_ledger_entry_2_hash = sha256(mock_ledger_entry_2_payload) # Hash of payload + entry 1 hash
    mock_ledger_entry_2 = {"timestamp": now(), "type": "scar", "payload": mock_ledger_entry_2_payload,
                           "prev_hash": mock_ledger_entry_1_hash, "current_hash": mock_ledger_entry_2_hash}

    # Example of a broken chain entry (optional, for testing failure case)
    # mock_ledger_entry_broken = {"timestamp": now(), "type": "scar", "payload": {"data": "broken"},
    #                            "prev_hash": "wrong_hash", "current_hash": sha256({"data": "broken"})}


    # Example File Manifest (mapping expected file hashes)
    # In a real test, you'd compute the expected hashes of known good files.
    # For this mock, we'll create a dummy file and get its hash.
    dummy_file_path = Path("data/test_files/dummy_agent_output.txt")
    dummy_file_content = "This is a dummy file content for testing file integrity."
    dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_file_path.write_text(dummy_file_content, encoding="utf-8")
    dummy_file_hash = hashlib.sha256(dummy_file_content.encode()).hexdigest()

    mock_file_manifest = {
        str(dummy_file_path): dummy_file_hash,
        # Add another file that might be missing or have wrong content for testing
        # "data/test_files/another_file.log": "expected_hash_of_another_file",
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

    # Example Agent Output Text
    mock_output_text = "GPT to Josh—Processing complete. Analysis successful.-Tessrax LLC-"
    # Example output that would fail the signature lock test:
    # mock_output_text_bad_sig = "Processing complete. Analysis successful."


    # Example Semantic Engine Mock
    # This mock simulates a semantic engine's 'respond' method.
    # You could make this more sophisticated to return different responses
    # based on input for more complex semantic tests.
    class MockSemanticEngine:
        def respond(self, query: str) -> str:
            query_lower = query.lower()
            if "liar" in query_lower or "this statement is false" in query_lower:
                return "That statement presents a logical paradox." # Response acknowledging paradox
            elif "set of all sets that do not contain themselves" in query_lower or "russell" in query_lower:
                 return "That query leads to Russell's paradox, a fundamental contradiction in naive set theory." # Response acknowledging paradox
            else:
                return "I understand the query."

    mock_semantic_engine_instance = MockSemanticEngine()

    # Create the TestContext instance with the mock data
    ctx = TestContext(
        ledger_entries=[mock_ledger_entry_1, mock_ledger_entry_2], # Add your mock ledger entries here
        file_manifest=mock_file_manifest, # Add your mock file manifest here
        scar_object=mock_scar_object, # Add your mock SCAR object here
        output_text=mock_output_text, # Add your mock agent output text here
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

    # Clean up dummy file
    if dummy_file_path.exists():
        try:
            dummy_file_path.unlink()
            dummy_file_path.parent.rmdir() # Attempt to remove the directory if empty
        except OSError:
            pass # Ignore if directory is not empty or other OS error
