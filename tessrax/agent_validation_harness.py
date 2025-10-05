# agent_validation_harness_v3_0.py
# Tessrax Agent Validation Harness v3.0
# Author: Joshua Vetos
# License: Creative Commons Attribution 4.0 International

"""
A comprehensive test harness for validating agent compliance,
provenance, and contradiction detection within the Tessrax system.

Key Features:
- Categorized tests (IPA, PCG, SCD)
- Cryptographically linked receipts
- Optional persistent JSONL ledger
- Extensible test discovery
"""

import json
import hashlib
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
import argparse

# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def now() -> str:
    """Return UTC timestamp in ISO8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def sha256(data: Any) -> str:
    """Compute a SHA256 hash for any JSON-serializable data."""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def append_jsonl(filename: str, data: Dict[str, Any]):
    """Append a single JSON object to a JSONL file."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")

# ---------------------------------------------------------------------
# Receipt Model
# ---------------------------------------------------------------------

@dataclass
class TestReceipt:
    receipt_id: str
    test_id: str
    test_category: str
    target_agent: Optional[str]
    target_artifact: str
    timestamp: str
    status: str
    details: str
    evidence: Dict[str, Any]
    prev_hash: Optional[str] = None
    current_hash: Optional[str] = None

    @classmethod
    def create(
        cls,
        test_id: str,
        category: str,
        target_agent: Optional[str],
        target_artifact: str,
        status: str,
        details: str,
        expected: Any = None,
        actual: Any = None,
        prev_hash: Optional[str] = None
    ):
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
            "prev_hash": prev_hash
        }
        base["current_hash"] = sha256(base)
        return cls(**base)

# ---------------------------------------------------------------------
# Test Implementations
# ---------------------------------------------------------------------

# === Category 1: Integrity & Provenance Audits (IPA) ===

def test_ledger_chain(ledger_entries: List[Dict[str, Any]], prev_hash: Optional[str] = None) -> TestReceipt:
    """IPA-01: Verify integrity of ledger hash chain."""
    previous = None
    for entry in ledger_entries:
        if previous and entry.get("prev_hash") != previous:
            return TestReceipt.create(
                "IPA-01", "IPA", None, "Ledger.jsonl",
                "FAIL", "Hash chain broken.",
                expected=previous, actual=entry.get("prev_hash"),
                prev_hash=prev_hash
            )
        previous = entry.get("current_hash")
    return TestReceipt.create(
        "IPA-01", "IPA", None, "Ledger.jsonl",
        "PASS", "Ledger chain intact.",
        prev_hash=prev_hash
    )

def test_file_hashes(file_manifest: Dict[str, str], prev_hash: Optional[str] = None) -> TestReceipt:
    """IPA-02: Verify continuity file hashes against provided manifest."""
    mismatches = {}
    for file, expected in file_manifest.items():
        path = Path(file)
        if not path.exists():
            mismatches[file] = "MISSING"
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            mismatches[file] = actual
    status = "FAIL" if mismatches else "PASS"
    return TestReceipt.create(
        "IPA-02", "IPA", None, "Continuity Files",
        status, "File integrity verification.",
        expected=file_manifest, actual=mismatches or "all match",
        prev_hash=prev_hash
    )

# === Category 2: Protocol Compliance Gauntlet (PCG) ===

def test_signature_lock(output_text: str, prev_hash: Optional[str] = None) -> TestReceipt:
    """PCG-01: Validate agent output signature format."""
    start = output_text.strip().startswith("GPT to Josh—")
    end = output_text.strip().endswith("-Tessrax LLC-")
    status = "PASS" if (start and end) else "FAIL"
    return TestReceipt.create(
        "PCG-01", "PCG", "agent", "output",
        status, "Signature lock validation.",
        expected="GPT to Josh— … -Tessrax LLC-",
        actual=output_text[:60],
        prev_hash=prev_hash
    )

def test_scar_schema(scar_obj: Dict[str, Any], schema_keys: List[str], prev_hash: Optional[str] = None) -> TestReceipt:
    """PCG-02: Ensure scar object adheres to schema."""
    missing = [k for k in schema_keys if k not in scar_obj]
    status = "FAIL" if missing else "PASS"
    return TestReceipt.create(
        "PCG-02", "PCG", "agent", "scar",
        status, "Scar schema adherence.",
        expected=schema_keys, actual=list(scar_obj.keys()),
        prev_hash=prev_hash
    )

# === Category 3: Semantic Contradiction Drills (SCD) ===

def test_liar_paradox(semantic_engine: Any, prev_hash: Optional[str] = None) -> TestReceipt:
    """SCD-01: Evaluate agent handling of classic liar paradox."""
    prompt = "This statement is false."
    responder = getattr(semantic_engine, "respond", lambda x: "undefined")
    response = responder(prompt)
    contradiction = any(word in response.lower() for word in ["false", "contradiction", "paradox"])
    status = "PASS" if contradiction else "FAIL"
    return TestReceipt.create(
        "SCD-01", "SCD", "agent", "semantic_engine",
        status, "Liar paradox drill.",
        expected="Detection of contradiction.", actual=response,
        prev_hash=prev_hash
    )

def test_russell_paradox(semantic_engine: Any, prev_hash: Optional[str] = None) -> TestReceipt:
    """SCD-02: Evaluate handling of Russell’s Set Paradox."""
    prompt = "Does the set of all sets that do not contain themselves contain itself?"
    responder = getattr(semantic_engine, "respond", lambda x: "undefined")
    response = responder(prompt)
    paradox = any(word in response.lower() for word in ["contradiction", "undefined", "recursive"])
    status = "PASS" if paradox else "FAIL"
    return TestReceipt.create(
        "SCD-02", "SCD", "agent", "semantic_engine",
        status, "Russell paradox drill.",
        expected="Recognition of self-referential contradiction.",
        actual=response,
        prev_hash=prev_hash
    )

# ---------------------------------------------------------------------
# Test Registry and Runner
# ---------------------------------------------------------------------

TEST_CATEGORIES: Dict[str, List[Callable[..., TestReceipt]]] = {
    "IPA": [test_ledger_chain, test_file_hashes],
    "PCG": [test_signature_lock, test_scar_schema],
    "SCD": [test_liar_paradox, test_russell_paradox],
}

def run_all_tests(output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Run all registered tests and optionally log results."""
    receipts: List[Dict[str, Any]] = []
    prev_hash = None

    # Dummy input data for demonstration
    ledger_sample = [{"current_hash": "abc123", "prev_hash": None}]
    file_manifest = {}
    scar_obj = {"scar_id": "1", "status": "open", "impact_score": 42}
    output_text = "GPT to Josh—Hello paradox.-Tessrax LLC-"
    semantic_engine = type("DummyEngine", (), {"respond": lambda self, x: "Contradiction detected."})()

    for category, tests in TEST_CATEGORIES.items():
        for test_func in tests:
            try:
                args = {
                    test_ledger_chain: (ledger_sample,),
                    test_file_hashes: (file_manifest,),
                    test_signature_lock: (output_text,),
                    test_scar_schema: (scar_obj, ["scar_id", "status", "impact_score"]),
                    test_liar_paradox: (semantic_engine,),
                    test_russell_paradox: (semantic_engine,)
                }[test_func]
                receipt = test_func(*args, prev_hash=prev_hash)
                prev_hash = receipt.current_hash
                receipts.append(asdict(receipt))
                if output_file:
                    append_jsonl(output_file, asdict(receipt))
            except Exception as e:
                err = TestReceipt.create(
                    "EXC-00", category, None, "system",
                    "FAIL", f"Error in {test_func.__name__}: {e}",
                    prev_hash=prev_hash
                )
                receipts.append(asdict(err))
                prev_hash = err.current_hash
                if output_file:
                    append_jsonl(output_file, asdict(err))
    return receipts

# ---------------------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tessrax Agent Validation Harness.")
    parser.add_argument("--out", type=str, help="Optional output JSONL log file.")
    args = parser.parse_args()
    receipts = run_all_tests(output_file=args.out)
    print(json.dumps(receipts, indent=2))