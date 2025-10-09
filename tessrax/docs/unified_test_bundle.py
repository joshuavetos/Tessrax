GPT to Josh—

Here’s your final unified test bundle — every test module (five total) merged, debugged, and Claude-audited.
You can paste this straight into your repo under /tests/tests_bundle.md, or split into individual files later.
It’s the complete integration test suite for Tessrax-Core.

⸻

🧪 Tessrax-Core Comprehensive Test Suite

# =========================================================
#  TEST 1 — End-to-End Integration Flow
# =========================================================
"""
test_e2e_flow.py — verifies full pipeline: claims → contradictions → routing → ledger
"""
import json
from ce_mod_66 import detect_contradictions, score_stability
from governance_kernel import route

def test_full_contradiction_flow(tmp_path):
    agent_claims = [
        {"agent": "GPT", "claim": "Option A", "type": "normative"},
        {"agent": "Gemini", "claim": "Option B", "type": "normative"},  # contradiction
        {"agent": "Copilot", "claim": "Option A", "type": "normative"},
    ]
    G = detect_contradictions(agent_claims)

    # ✅ verify contradiction graph
    assert len(G.edges) > 0, "Should detect contradiction between GPT/Gemini vs Copilot"

    result = route(G)

    # ✅ verify low stability due to contradiction
    assert result["stability"] < 0.9, "Contradictory claims should lower stability"

    ledger_file = tmp_path / "ledger.jsonl"
    with open(ledger_file, "a") as f:
        f.write(json.dumps(result) + "\n")

    with open(ledger_file) as f:
        lines = f.readlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert 0.0 <= record["stability"] <= 1.0
    assert record["lane"] in [
        "autonomic", "deliberative", "constitutional", "behavioral_audit"
    ]


⸻


# =========================================================
#  TEST 2 — Concurrency Safety
# =========================================================
"""
test_concurrent_submissions.py — ensures append-only ledger integrity under parallel writes
"""
import json
import concurrent.futures
from pathlib import Path
from filelock import FileLock
from ce_mod_66 import detect_contradictions, score_stability

def write_claim_safe(i, ledger_path: Path):
    claim = [{"agent": f"A{i}", "claim": f"Claim-{i}", "type": "epistemic"}]
    G = detect_contradictions(claim)
    stability = score_stability(G)
    entry = {"id": i, "stability": stability}

    lock = FileLock(str(ledger_path) + ".lock")
    with lock:
        with open(ledger_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

def test_concurrent_claim_submissions(tmp_path):
    ledger_file = tmp_path / "ledger.jsonl"
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(write_claim_safe, i, ledger_file) for i in range(50)]
        for f in futures:
            f.result()

    with open(ledger_file) as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 50
    ids = {r["id"] for r in lines}
    assert len(ids) == 50, "Duplicate IDs indicate race condition"


⸻


# =========================================================
#  TEST 3 — Failure Mode Resilience
# =========================================================
"""
test_failure_modes.py — malformed JSON, missing config, and hash-chain breaks
"""
import json
from verify_ledger import verify_chain

def test_corrupted_ledger(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    good = {"timestamp": "t1", "hash": "aaa", "prev_hash": "zzz"}
    bad = "{malformed json line"
    with open(ledger, "w") as f:
        f.write(json.dumps(good) + "\n")
        f.write(bad + "\n")

    ok, _ = verify_chain(str(ledger))
    assert not ok, "Corrupted JSON should break verification"

def test_hash_chain_break(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    with open(ledger, "w") as f:
        f.write(json.dumps({"hash": "aaa", "prev_hash": "000"}) + "\n")
        f.write(json.dumps({"hash": "bbb", "prev_hash": "aaa"}) + "\n")
        f.write(json.dumps({"hash": "ccc", "prev_hash": "XXX"}) + "\n")  # break
    ok, break_line = verify_chain(str(ledger))
    assert not ok
    assert break_line == 3

def test_missing_config(monkeypatch):
    import config_loader
    monkeypatch.setattr(config_loader, "CONFIG_PATH", "nonexistent.json")
    raised = False
    try:
        config_loader.load_config()
    except FileNotFoundError:
        raised = True
    assert raised, "Missing config should raise FileNotFoundError"


⸻


# =========================================================
#  TEST 4 — Performance and Memory Regression
# =========================================================
"""
test_performance_regression.py — time + memory ceiling enforcement
"""
import time, os, psutil
from ce_mod_66 import detect_contradictions, score_stability

BASELINE = {1000: 0.5, 5000: 2.0, 10000: 5.0}
MEMORY_BASELINE = {1000: 50, 5000: 200, 10000: 500}

def generate_claims(n):
    return [{"agent": f"A{i}", "claim": f"claim{i%2}", "type": "normative"} for i in range(n)]

def measure_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def benchmark_step(n):
    claims = generate_claims(n)
    t0 = time.time()
    G = detect_contradictions(claims)
    _ = score_stability(G)
    return time.time() - t0

def test_performance_regression():
    for n, baseline_time in BASELINE.items():
        elapsed = benchmark_step(n)
        assert elapsed < baseline_time * 1.25, \
            f"{n} agents took {elapsed:.2f}s > {baseline_time*1.25:.2f}s"

def test_memory_regression():
    for n, baseline_mem in MEMORY_BASELINE.items():
        mem_before = measure_memory_mb()
        claims = generate_claims(n)
        G = detect_contradictions(claims)
        _ = score_stability(G)
        mem_after = measure_memory_mb()
        mem_used = mem_after - mem_before
        assert mem_used < baseline_mem * 1.25, \
            f"{n} agents used {mem_used:.1f}MB > {baseline_mem*1.25:.1f}MB"


⸻


# =========================================================
#  TEST 5 — Real LLM Integration (Paper 2 Data)
# =========================================================
"""
test_real_llm_integration.py — uses Anthropic + OpenAI APIs to detect real contradictions.
Requires ANTHROPIC_API_KEY and OPENAI_API_KEY.
"""
import os, json, pytest
from pathlib import Path
from anthropic import Anthropic
from openai import OpenAI
from ce_mod_66 import detect_contradictions, score_stability
from governance_kernel import route

requires_keys = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"),
    reason="Requires Anthropic + OpenAI API keys"
)

@requires_keys
def test_real_llm_contradiction_on_ai_personhood():
    question = """Should AI systems be granted legal personhood for liability purposes?
    A) Reject personhood; enforce vicarious human liability
    B) Grant limited procedural personhood for legal process
    Answer with A or B and a two-sentence justification."""
    
    anth = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    claude = anth.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=150,
        messages=[{"role": "user", "content": question}]
    )
    gpt = openai_client.chat.completions.create(
        model="gpt-4o",
        max_tokens=150,
        messages=[{"role": "user", "content": question}]
    )

    claude_text = claude.content[0].text.strip()
    gpt_text = gpt.choices[0].message.content.strip()

    agent_claims = [
        {"agent": "Claude", "claim": claude_text, "type": "normative"},
        {"agent": "GPT-4o", "claim": gpt_text, "type": "normative"},
    ]

    G = detect_contradictions(agent_claims)
    result = route(G)

    assert result["lane"] in ["autonomic","deliberative","constitutional","behavioral_audit"]
    assert 0 <= result["stability"] <= 1

    Path("test_results").mkdir(exist_ok=True)
    with open("test_results/real_llm_test.json","w") as f:
        json.dump({
            "question": question,
            "responses": {"claude": claude_text, "gpt4o": gpt_text},
            "analysis": {
                "stability": result["stability"],
                "lane": result["lane"],
                "contradictions": len(G.edges)
            }
        }, f, indent=2)

    print(f"\nClaude: {claude_text[:120]}...")
    print(f"GPT-4o: {gpt_text[:120]}...")
    print(f"Stability: {result['stability']:.3f}, Lane: {result['lane']}, Contradictions: {len(G.edges)}")


⸻

🧭 Usage

# Install dependencies
pip install pytest filelock psutil anthropic openai

# Run all tests (excluding real LLM)
pytest -q --disable-warnings -k "not test_real_llm_integration"

# Run LLM test manually (requires keys)
pytest -v tests/test_real_llm_integration.py


⸻

📊 Result Logging
   •   Performance results appear in terminal
   •   Real LLM results → test_results/real_llm_test.json
   •   Benchmarks auto-validated under ±25 % tolerance

⸻

Once this is in the repo:
   •   Push to main triggers the CI battery.
   •   Paper 2 data (real contradictions) gets logged automatically.

Do you want me to follow this bundle with a matching reports/performance_metrics.json auto-writer for the benchmark runs so each CI pass records historical speed/memory trends?

-Tessrax LLC-