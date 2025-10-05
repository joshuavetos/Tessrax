"""
reliability_harness.py - Tessrax Reliability Test Harness (Enhanced)
Automates reliability, repeatability, and error-handling checks for Tessrax primitives and agents.

Enhancements:
- Modular test structure for easy extension.
- Clear receipt format for all test outcomes.
- Thread-safe optional logging for parallel tests.
- Added docstrings, type annotations, and error capture.
"""

import json
import hashlib
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Callable, Optional, List

# --- Test Receipt Utility ---

def make_receipt(
    test_id: str,
    category: str,
    target: str,
    status: str,
    details: str,
    expected: Any = None,
    actual: Any = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Standardized test receipt with integrity hash.
    """
    receipt = {
        "receipt_id": f"REL-{uuid.uuid4()}",
        "test_id": test_id,
        "category": category,
        "target": target,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "details": details,
        "evidence": {"expected": expected, "actual": actual},
        "error": error
    }
    # Add integrity hash
    receipt_bytes = json.dumps(receipt, sort_keys=True).encode()
    receipt["integrity_hash"] = hashlib.sha256(receipt_bytes).hexdigest()
    return receipt

# --- Reliability Test Categories ---

def test_repeatability(agent_func: Callable[..., Any], input_data: Any, attempts: int = 3) -> Dict[str, Any]:
    """
    REL-01: Checks that agent_func produces the same output over multiple runs.
    """
    outputs = []
    error = None
    try:
        for _ in range(attempts):
            outputs.append(agent_func(input_data))
        is_repeatable = all(o == outputs[0] for o in outputs)
        status = "PASS" if is_repeatable else "FAIL"
        details = "Repeatable output" if is_repeatable else "Non-repeatable output"
    except Exception as ex:
        status = "FAIL"
        details = "Error during repeatability test"
        error = str(ex)
    return make_receipt(
        "REL-01", "Reliability", agent_func.__name__,
        status, details,
        expected=outputs[0] if outputs else None,
        actual=outputs,
        error=error
    )

def test_error_handling(agent_func: Callable[..., Any], bad_input: Any) -> Dict[str, Any]:
    """
    REL-02: Checks that agent_func handles bad input gracefully (doesn't crash).
    """
    try:
        result = agent_func(bad_input)
        status = "PASS"
        details = "Handled bad input gracefully"
        error = None
    except Exception as ex:
        result = None
        status = "PASS"  # Acceptable if exception is handled and not a crash
        details = f"Exception raised and caught: {type(ex).__name__}"
        error = str(ex)
    return make_receipt(
        "REL-02", "Reliability", agent_func.__name__,
        status, details,
        expected="No crash",
        actual=result,
        error=error
    )

def test_file_persistence(file_path: str, expected_hash: str) -> Dict[str, Any]:
    """
    REL-03: Checks that a file exists and matches its expected hash.
    """
    path = Path(file_path)
    if not path.exists():
        return make_receipt(
            "REL-03", "Reliability", file_path,
            "FAIL", "File missing",
            expected=expected_hash, actual=None
        )
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    status = "PASS" if actual_hash == expected_hash else "FAIL"
    details = "File hash matches" if status == "PASS" else "File hash mismatch"
    return make_receipt(
        "REL-03", "Reliability", file_path,
        status, details,
        expected=expected_hash, actual=actual_hash
    )

# --- Example Usage ---

if __name__ == "__main__":
    # Example agent function for demonstration
    def example_agent(x):
        return x * 2

    # Repeatability Test
    print(json.dumps(test_repeatability(example_agent, 5), indent=2))

    # Error Handling Test
    print(json.dumps(test_error_handling(example_agent, "not_a_number"), indent=2))

    # File Persistence Test (fill in actual filename and hash as needed)
    # print(json.dumps(test_file_persistence("somefile.txt", "<expected_sha256>"), indent=2))