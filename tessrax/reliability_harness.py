"""
Tessrax Reliability Harness v2.0
---------------------------------
Self-auditing code validation pipeline for AI-generated code.

Workflow:
1. Execute and validate AI-generated code in a sandbox.
2. Generate a signed receipt attesting to the results.
3. Append the receipt to the Tessrax ledger (tamper-evident).

Dependencies:
- tessrax/core/receipts.py
- tessrax/core/ledger.py
- pytest, mypy installed
"""

import tempfile
import subprocess
import json
import hashlib
import os
import time
from pathlib import Path
from typing import Dict, Any

from tessrax.core.receipts import create_receipt
from tessrax.core.ledger import SQLiteLedger
from nacl.signing import SigningKey
from nacl.encoding import HexEncoder


PYTHON = "python3"


# ---------------------------------------------------------------------
# Utility: run subprocess safely
# ---------------------------------------------------------------------

def run_cmd(cmd: list[str], timeout: int = 15) -> Dict[str, Any]:
    """Run a subprocess with timeout, capturing stdout/stderr/exit."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": " ".join(cmd),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"cmd": " ".join(cmd), "error": "timeout"}
    except Exception as e:
        return {"cmd": " ".join(cmd), "error": str(e)}


# ---------------------------------------------------------------------
# Core: Validate AI code
# ---------------------------------------------------------------------

def validate_ai_code(code_str: str, tests_str: str) -> Dict[str, Any]:
    """Execute code + tests and return structured results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = Path(tmpdir) / "ai_code.py"
        tests_path = Path(tmpdir) / "test_ai_code.py"

        code_path.write_text(code_str, encoding="utf-8")
        tests_path.write_text(tests_str, encoding="utf-8")

        results = {"steps": [], "valid": False}

        # 1. Syntax
        syntax = run_cmd([PYTHON, "-m", "py_compile", str(code_path)])
        results["steps"].append({"phase": "syntax", **syntax})
        if syntax.get("exit_code", 1) != 0:
            results["valid"] = False
            results["reason"] = "syntax_error"
            return results

        # 2. Type check
        type_check = run_cmd(["mypy", "--ignore-missing-imports", str(code_path)])
        results["steps"].append({"phase": "mypy", **type_check})

        # 3. Run tests
        test_run = run_cmd(["pytest", "-q", str(tests_path)])
        results["steps"].append({"phase": "pytest", **test_run})

        # Final verdict
        if test_run.get("exit_code") == 0:
            results["valid"] = True
        else:
            results["valid"] = False
            results["reason"] = "test_failure"

        return results


# ---------------------------------------------------------------------
# Signing + Ledger Integration
# ---------------------------------------------------------------------

def attest_validation(
    code_str: str,
    tests_str: str,
    verdict: Dict[str, Any],
    signing_key_hex: str,
    ledger: SQLiteLedger,
) -> Dict[str, Any]:
    """Generate signed receipt + commit to ledger."""
    # Canonicalize and hash materials
    code_hash = hashlib.sha256(code_str.encode()).hexdigest()
    test_hash = hashlib.sha256(tests_str.encode()).hexdigest()
    verdict_json = json.dumps(verdict, sort_keys=True)
    verdict_hash = hashlib.sha256(verdict_json.encode()).hexdigest()

    payload = {
        "entry_type": "validation_receipt",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_hash": code_hash,
        "test_hash": test_hash,
        "verdict_hash": verdict_hash,
        "verdict": verdict,
    }

    # Sign receipt
    receipt = create_receipt(private_key_hex=signing_key_hex, event_payload=payload)
    # Commit to ledger
    ledger.add_event({"entry_type": "receipt", "timestamp": payload["timestamp"], "receipt": receipt})
    return receipt


# ---------------------------------------------------------------------
# Demo execution (self-contained)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Example: trivial function + tests
    ai_code = '''
def add(a: int, b: int) -> int:
    return a + b
'''

    test_spec = '''
from ai_code import add

def test_add_basic():
    assert add(2,3) == 5

def test_add_negative():
    assert add(-1,2) == 1
'''

    # Run validation
    verdict = validate_ai_code(ai_code, test_spec)
    print("[Harness] Validation complete.")
    print(json.dumps(verdict, indent=2))

    # Sign + record
    key = SigningKey.generate()
    signing_key_hex = key.encode(encoder=HexEncoder).decode("utf-8")
    ledger = SQLiteLedger()

    receipt = attest_validation(ai_code, test_spec, verdict, signing_key_hex, ledger)
    print("\n[Harness] Signed receipt:")
    print(json.dumps(receipt, indent=2))

    print("\n[Harness] Ledger stats:")
    print(json.dumps(ledger.stats(), indent=2))