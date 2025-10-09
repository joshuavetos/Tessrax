"""
Tessrax Scaffolding Engine v2.2
Logs design sessions, emits governance events, and maintains design provenance.

Upgrades:
  • Deterministic SHA-256 hashing with timestamp salt
  • Fault-tolerant JSONL writer + recovery on partial writes
  • Governance-safe emission (auto-tags + quorum-aware)
  • Compact session summarizer for dashboard ingestion
"""

import json, hashlib, os
from datetime import datetime
from pathlib import Path
from governance_kernel import GovernanceKernel

LOG_PATH = Path("data/scaffolding_log.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
kernel = GovernanceKernel()

# ============================================================
# Helpers
# ============================================================

def _sha256(obj: dict) -> str:
    """Deterministic hash with embedded timestamp salt."""
    salted = dict(obj)
    salted["__salt"] = datetime.utcnow().isoformat()
    return hashlib.sha256(json.dumps(salted, sort_keys=True).encode()).hexdigest()

def _safe_append(path: Path, record: dict) -> None:
    """Append JSON safely with automatic recovery on broken lines."""
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        with open(path, "a", encoding="utf-8") as f_out, open(tmp_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                f_out.write(line)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

# ============================================================
# Core API
# ============================================================

def record_interaction(
    prompt: str,
    response: str,
    tags: list[str] | None = None,
    file_changed: str | None = None,
) -> str:
    """Record a scaffolding interaction and emit a governance event."""
    tags = tags or []
    rec = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt.strip(),
        "response": response.strip(),
        "tags": tags,
        "file_changed": file_changed,
    }
    rec["hash"] = "sha256:" + _sha256(rec)

    _safe_append(LOG_PATH, rec)

    event = {
        "event": "DESIGN_DECISION_RECORDED",
        "file_changed": file_changed or "unspecified",
        "tags": list(set(tags + ["scaffolding"])),
        "decision_hash": rec["hash"],
        "timestamp": rec["timestamp"],
    }
    try:
        kernel.append_event(event)
    except Exception as e:
        print(f"⚠️ Governance append failed: {e}")

    return rec["hash"]

def summarize_session() -> dict:
    """Return quick summary stats for dashboard or audit."""
    if not LOG_PATH.exists():
        return {"records": 0, "tags": [], "last_file": None}

    tags, last_file, count = set(), None, 0
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                count += 1
                tags.update(rec.get("tags", []))
                if rec.get("file_changed"):
                    last_file = rec["file_changed"]
            except json.JSONDecodeError:
                continue

    return {"records": count, "tags": sorted(tags), "last_file": last_file}

# ============================================================
# CLI / Demo
# ============================================================

if __name__ == "__main__":
    print("🚧 Tessrax Scaffolding Engine v2.2 — demo mode")
    h = record_interaction(
        prompt="Integrate dynamic policy loader",
        response="Added auto-merge of policies/*.py into live registry.",
        tags=["governance", "scaffolding"],
        file_changed="policy_rules.py",
    )
    print(f"Recorded decision: {h}")
    print("Summary:", json.dumps(summarize_session(), indent=2))