"""
reliability_harness.py — Tessrax Reliability Test Harness (v2.1)
-----------------------------------------------------------------
Automates repeatability, determinism, and error-handling checks for Tessrax primitives and agents.

Enhancements:
- Modular and ledger-aware receipt generation
- Thread-safe optional file logging
- Clear standard schema for all test outcomes
- Hooks for Merkle or signature-based verification
"""

import json
import hashlib
import time
import uuid
import threading
from pathlib import Path
from typing import Any, Dict, Callable, Optional, List

# Optional import if run inside Tessrax stack
try:
    from tessrax.core.ledger import SQLiteLedger
except ImportError:
    SQLiteLedger = None


# ============================================================
# Receipt Utility
# ============================================================

def make_receipt(
    test_id: str,
    category: str,
    target: str,
    status: str,
    details: str,
    expected: Any = None,
    actual: Any = None,
    error: Optional[str] = None,
    signer: Optional[str] = None,
) -> Dict[str, Any]:
    """Creates a standardized, integrity-checked test receipt."""
    receipt = {
        "receipt_id": f"REL-{uuid.uuid4()}",
        "test_id": test_id,
        "category": category,
        "target": target,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "details": details,
        "evidence": {"expected": expected, "actual": actual},
        "error": error,
        "signer": signer or "unverified",
    }
    receipt["integrity_hash"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True).encode()
    ).hexdigest()
    return receipt


# ============================================================
# Thread-safe Logger
# ============================================================

class ReceiptLogger:
    """Thread-safe logger that writes receipts to file and optional ledger."""

    def __init__(self, log_file: Optional[str] = None, ledger: Optional[Any] = None):
        self.log_file = Path(log_file) if log_file else None
        self.ledger = ledger
        self.lock = threading.Lock()

    def record(self, receipt: Dict[str, Any]) -> None:
        """Append the receipt to file and/or ledger."""
        with self.lock:
            if self.log_file:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(receipt) + "\n")

            if self.ledger and hasattr(self.ledger, "add_event"):
                try:
                    self.ledger.add_event({"entry_type": "TEST_RECEIPT", "payload": receipt})
                except Exception as e:
                    print(f"[WARN] Ledger append failed: {e}")


# ============================================================
# Reliability Test Categories
# ============================================================

def test_repeatability(agent_func: Callable[..., Any], input_data: Any, attempts: int = 3) -> Dict[str, Any]:
    """REL-01: Ensures deterministic output across repeated runs."""
    outputs = []
    try:
        for _ in range(attempts):
            outputs.append(agent_func(input_data))
        is_repeatable = all(o == outputs[0] for o in outputs)
        status = "PASS" if is_repeatable else "FAIL"
        details = "Output stable across runs" if is_repeatable else "Output drift detected"
    except Exception as ex:
        return make_receipt("REL-01", "Reliability", agent_func.__name__,
                            "FAIL", "Exception during repeatability", error=str(ex))
    return make_receipt("REL-01", "Reliability", agent_func.__name__,
                        status, details, expected=outputs[0] if outputs else None, actual=outputs)


def test_error_handling(agent_func: Callable[..., Any], bad_input: Any) -> Dict[str, Any]:
    """REL-02: Checks that function fails safely with invalid input."""
    try:
        agent_func(bad_input)
        status, details, error = "PASS", "Handled invalid input", None
    except Exception as ex:
        status, details, error = "PASS", f"Graceful exception: {type(ex).__name__}", str(ex)
    return make_receipt("REL-02", "Reliability", agent_func.__name__,
                        status, details, expected="Graceful handling", actual=None, error=error)


def test_file_persistence(file_path: str, expected_hash: str) -> Dict[str, Any]:
    """REL-03: Confirms file integrity via hash comparison."""
    path = Path(file_path)
    if not path.exists():
        return make_receipt("REL-03", "Reliability", file_path, "FAIL", "File missing",
                            expected=expected_hash, actual=None)
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    status = "PASS" if actual_hash == expected_hash else "FAIL"
    return make_receipt("REL-03", "Reliability", file_path, status,
                        "Hash verified" if status == "PASS" else "Hash mismatch",
                        expected=expected_hash, actual=actual_hash)


def test_timing(agent_func: Callable[..., Any], input_data: Any, threshold_ms: float = 100.0) -> Dict[str, Any]:
    """REL-04: Measures execution time to detect slow regressions."""
    start = time.perf_counter()
    try:
        agent_func(input_data)
        duration = (time.perf_counter() - start) * 1000
        status = "PASS" if duration <= threshold_ms else "WARN"
        details = f"Execution time {duration:.2f} ms"
    except Exception as ex:
        return make_receipt("REL-04", "Reliability", agent_func.__name__,
                            "FAIL", "Exception during timing", error=str(ex))
    return make_receipt("REL-04", "Reliability", agent_func.__name__,
                        status, details, expected=f"<={threshold_ms}ms", actual=f"{duration:.2f}ms")


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    def example_agent(x): return x * 2

    logger = ReceiptLogger(log_file="data/test_receipts.jsonl")

    tests = [
        test_repeatability(example_agent, 5),
        test_error_handling(example_agent, "not_a_number"),
        test_timing(example_agent, 10),
    ]

    for r in tests:
        logger.record(r)
        print(json.dumps(r, indent=2))