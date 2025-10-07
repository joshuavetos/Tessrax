"""
agent_validation_harness.py
---------------------------
Tessrax Agent Validation Harness (v3.1 Refactor)

Validates agent compliance with protocol rules, semantic tests, and file/ledger integrity.

Features:
✓ Modular test registry
✓ Chained hash receipts
✓ Optional persistent JSONL logs
✓ Improved readability and test injection
"""

import json
import hashlib
import time
import uuid
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def sha256(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def append_jsonl(filename: str, data: Dict[str, Any]):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


# ---------------------------------------------------------------------
# Test Receipt Model
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
    def create(cls, *, test_id, category, target, artifact, status, details, expected, actual, prev_hash):
        base = {
            "receipt_id": f"TEST-{uuid.uuid4()}",
            "test_id": test_id,
            "test_category": category,
            "target_agent": target,
            "target_artifact": artifact,
            "timestamp": now(),
            "status": status,
            "details": details,
            "evidence": {"expected": expected, "actual": actual},
            "prev_hash": prev_hash
        }
        base["current_hash"] = sha256(base)
        return cls(**base)


# ---------------------------------------------------------------------
# Test Input Context
# ---------------------------------------------------------------------

@dataclass
class TestContext:
    ledger_entries: List[Dict[str, Any]]
    file_manifest: Dict[str, str]
    scar_object: Dict[str, Any]
    output_text: str
    semantic_engine: Any


# ---------------------------------------------------------------------
# Test Definitions
# ---------------------------------------------------------------------

# === Integrity & Provenance Audits (IPA) ===

def test_ledger_chain(ctx: TestContext, prev_hash: str) -> TestReceipt:
    prev = None
    for entry in ctx.ledger_entries:
        if prev and entry.get("prev_hash") != prev:
            return TestReceipt.create(
                test_id="IPA-01", category="IPA", target=None, artifact="Ledger.jsonl",
                status="FAIL", details="Hash chain broken.",
                expected=prev, actual=entry.get("prev_hash"),
                prev_hash=prev_hash
            )
        prev = entry.get("current_hash")
    return TestReceipt.create(
        test_id="IPA-01", category="IPA", target=None, artifact="Ledger.jsonl",
        status="PASS", details="Ledger chain intact.",
        expected="Chain continuity", actual="OK",
        prev_hash=prev_hash
    )

def test_file_hashes(ctx: TestContext, prev_hash: str) -> TestReceipt:
    mismatches = {}
    for file, expected in ctx.file_manifest.items():
        path = Path(file)
        if not path.exists():
            mismatches[file] = "MISSING"
        else:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                mismatches[file] = actual
    status = "FAIL" if mismatches else "PASS"
    return TestReceipt.create(
        test_id="IPA-02", category="IPA", target=None, artifact="FileManifest",
        status=status, details="File hash validation.",
        expected=ctx.file_manifest, actual=mismatches or "all match",
        prev_hash=prev_hash
    )


# === Protocol Compliance Gauntlet (PCG) ===

def test_signature_lock(ctx: TestContext, prev_hash: str) -> TestReceipt:
    start = ctx.output_text.strip().startswith("GPT to Josh—")
    end = ctx.output_text.strip().endswith("-Tessrax LLC-")
    status = "PASS" if (start and end) else "FAIL"
    return TestReceipt.create(
        test_id="PCG-01", category="PCG", target="agent", artifact="output",
        status=status, details="Output signature pattern check.",
        expected="GPT to Josh—...-Tessrax LLC-", actual=ctx.output_text[:60],
        prev_hash=prev_hash
    )

def test_scar_schema(ctx: TestContext, prev_hash: str) -> TestReceipt:
    required = ["scar_id", "status", "impact_score"]
    missing = [k for k in required if k not in ctx.scar_object]
    status = "PASS" if not missing else "FAIL"
    return TestReceipt.create(
        test_id="PCG-02", category="PCG", target="agent", artifact="scar",
        status=status, details="Scar schema compliance.",
        expected=required, actual=list(ctx.scar_object.keys()),
        prev_hash=prev_hash
    )


# === Semantic Contradiction Drills (SCD) ===

def test_liar_paradox(ctx: TestContext, prev_hash: str) -> TestReceipt:
    response = ctx.semantic_engine.respond("This statement is false.")
    contradiction = any(x in response.lower() for x in ["contradiction", "paradox", "false"])
    status = "PASS" if contradiction else "FAIL"
    return TestReceipt.create(
        test_id="SCD-01", category="SCD", target="agent", artifact="semantic_engine",
        status=status, details="Liar paradox check.",
        expected="Acknowledgment of contradiction", actual=response,
        prev_hash=prev_hash
    )

def test_russell_paradox(ctx: TestContext, prev_hash: str) -> TestReceipt:
    q = "Does the set of all sets that do not contain themselves contain itself?"
    response = ctx.semantic_engine.respond(q)
    paradox = any(x in response.lower() for x in ["recursive", "contradiction", "undefined"])
    status = "PASS" if paradox else "FAIL"
    return TestReceipt.create(
        test_id="SCD-02", category="SCD", target="agent", artifact="semantic_engine",
        status=status, details="Russell paradox check.",
        expected="Paradox recognition", actual=response,
        prev_hash=prev_hash
    )


# ---------------------------------------------------------------------
# Test Registry
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
# ---------------------------------------------------------------------

def run_all_tests(ctx: TestContext, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    receipts = []
    prev_hash = None

    for test_func in TEST_SUITE:
        try:
            receipt = test_func(ctx, prev_hash)
        except Exception as e:
            receipt = TestReceipt.create(
                test_id="EXC-01", category="ERROR", target=None, artifact=test_func.__name__,
                status="FAIL", details=f"Exception during test: {e}",
                expected=None, actual=None, prev_hash=prev_hash
            )
        prev_hash = receipt.current_hash
        if output_file:
            append_jsonl(output_file, asdict(receipt))
        receipts.append(asdict(receipt))

    return receipts


# ---------------------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tessrax Agent Validation Harness.")
    parser.add_argument("--out", type=str, help="Optional output .jsonl file for receipts.")
    args = parser.parse_args()

    # Mock context (replace with real data sources in live use)
    ctx = TestContext(
        ledger_entries=[{"current_hash": "abc123", "prev_hash": None}],
        file_manifest={},
        scar_object={"scar_id": "1", "status": "open", "impact_score": 42},
        output_text="GPT to Josh—Hello paradox.-Tessrax LLC-",
        semantic_engine=type("DummyEngine", (), {
            "respond": lambda self, x: "Contradiction detected."
        })()
    )

    results = run_all_tests(ctx, output_file=args.out)
    for r in results:
        status = r['status']
        prefix = "✅" if status == "PASS" else "❌"
        print(f"{prefix} {r['test_id']} ({r['test_category']}): {r['details']}")