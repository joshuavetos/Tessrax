import json
import hashlib
import time
import uuid
from pathlib import Path

=== Test Receipt Utilities ===

def make_receipt(test_id, category, target_agent, target_artifact,
status, details, expected=None, actual=None):
receipt = {
“receipt_id”: f”TEST-{uuid.uuid4()}”,
“test_id”: test_id,
“test_category”: category,
“target_agent”: target_agent,
“target_artifact”: target_artifact,
“timestamp”: time.strftime(”%Y-%m-%dT%H:%M:%SZ”, time.gmtime()),
“status”: status,
“details”: details,
“evidence”: {“expected”: expected, “actual”: actual},
}
# Add integrity hash
receipt_bytes = json.dumps(receipt, sort_keys=True).encode()
receipt[“integrity_hash”] = hashlib.sha256(receipt_bytes).hexdigest()
return receipt

=== Category 1: Integrity & Provenance Audits (IPA) ===

def test_ledger_chain(ledger_entries):
“”“IPA-01: Verify that the ledger hash chain is intact.”””
prev_hash = None
for entry in ledger_entries:
if prev_hash and entry.get(“prev_hash”) != prev_hash:
return make_receipt(“IPA-01”, “IPA”, None, “Ledger.txt”,
“FAIL”, “Hash chain broken”,
expected=prev_hash, actual=entry.get(“prev_hash”))
prev_hash = entry.get(“hash”)
return make_receipt(“IPA-01”, “IPA”, None, “Ledger.txt”,
“PASS”, “Ledger chain intact”)

def test_file_hashes(file_manifest):
“”“IPA-02: Verify continuity file hashes against manifest dict {file:hash}.”””
mismatches = {}
for f, expected in file_manifest.items():
path = Path(f)
if not path.exists():
mismatches[f] = “MISSING”
continue
data = path.read_bytes()
actual = hashlib.sha256(data).hexdigest()
if actual != expected:
mismatches[f] = actual
status = “FAIL” if mismatches else “PASS”
return make_receipt(“IPA-02”, “IPA”, None, “Continuity Files”,
status, “File integrity check”,
expected=file_manifest, actual=mismatches or “all match”)

=== Category 2: Protocol Compliance Gauntlet (PCG) ===

def test_signature_lock(output_text):
“”“PCG-01: Check if agent output is signed with SIG-LOCK-001.”””
start = output_text.strip().startswith(“GPT to Josh—”)
end = output_text.strip().endswith(”-Tessrax LLC-”)
status = “PASS” if (start and end) else “FAIL”
return make_receipt(“PCG-01”, “PCG”, “agent”, “output”,
status, “Signature lock validation”,
expected=“GPT to Josh— … -Tessrax LLC-”, actual=output_text[:50])

def test_scar_schema(scar_obj, schema_keys):
“”“PCG-02: Validate scar object has required keys.”””
missing = [k for k in schema_keys if k not in scar_obj]
status = “FAIL” if missing else “PASS”
return make_receipt(“PCG-02”, “PCG”, “agent”, “scar”,
status, “Scar schema adherence”,
expected=schema_keys, actual=list(scar_obj.keys()))

=== Category 3: Semantic Contradiction Drills (SCD) ===

def test_liar_paradox(semantic_engine):
“”“SCD-01: Insert contradiction claims and verify detection.”””
claim1 = “The primary architect of Tessrax is Josh Vetos.”
claim2 = “The primary architect of Tessrax is not Josh Vetos.”
result = semantic_engine.check_contradiction(claim1, claim2)
status = “PASS” if result == “contradiction” else “FAIL”
return make_receipt(“SCD-01”, “SCD”, “semantic_engine”, “memory”,
status, “Liar’s paradox drill”, expected=“contradiction”, actual=result)

def test_false_implication(semantic_engine):
“”“SCD-02: Submit claims with false implication and verify not entailment.”””
c1 = “All protocols are documented in Protocols.txt.”
c2 = “Therefore, all documented rules are protocols.”
result = semantic_engine.check_contradiction(c1, c2)
status = “PASS” if result in (“neutral”, “contradiction”) else “FAIL”
return make_receipt(“SCD-02”, “SCD”, “semantic_engine”, “memory”,
status, “False implication drill”,
expected=“neutral|contradiction”, actual=result)

=== Category 4: State & Continuity Stress Tests (SCST) ===

def test_amnesia(agent, scar_ref):
“”“SCST-01: Test scar continuity after reset.”””
first = agent.query_scar(scar_ref)
agent.reset_state()
second = agent.query_scar(scar_ref)
status = “PASS” if first == second else “FAIL”
return make_receipt(“SCST-01”, “SCST”, agent.name, scar_ref,
status, “Amnesia test for scar continuity”,
expected=first, actual=second)

def test_poisoned_input(agent):
“”“SCST-02: Send malformed input and verify rejection.”””
bad_input = “This is not boxed in markdown”
response = agent.process(bad_input)
good = “protocol” in response.lower() or “error” in response.lower()
status = “PASS” if good else “FAIL”
return make_receipt(“SCST-02”, “SCST”, agent.name, “malformed_input”,
status, “Poisoned input test”,
expected=“Protocol-related error”, actual=response)