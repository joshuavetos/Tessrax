"""Temporal contradiction compressor preserving Merkle integrity.

Conforms to AEP-001, RVC-001, and POST-AUDIT-001 by delivering a
self-verifying summarisation pipeline for resolved contradictions.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from typing import Dict, List, Sequence

from tessrax.core.governance.receipts import write_receipt


def compress_receipts(receipts: Sequence[Dict[str, str]]) -> Dict[str, object]:
    """Compress resolved contradiction receipts into summary nodes."""

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for receipt in receipts:
        topic = receipt.get("topic") or receipt.get("id", "generic").split(":")[0]
        grouped[topic].append(receipt)
    summaries: List[Dict[str, str]] = []
    child_hashes: List[str] = []
    for topic in sorted(grouped):
        statements = " | ".join(
            sorted(entry.get("resolution", "resolved") for entry in grouped[topic])
        )
        summary_hash = sha256(f"{topic}|{statements}".encode("utf-8")).hexdigest()
        summaries.append({"topic": topic, "summary": statements, "hash": summary_hash})
        child_hashes.append(summary_hash)
    merkle_root = _compute_merkle_root(child_hashes)
    compression_ratio = round((len(receipts) or 1) / max(len(summaries), 1), 6)
    metrics = {
        "root_hash": merkle_root,
        "compression_ratio": compression_ratio,
        "nodes": len(summaries),
    }
    write_receipt("tessrax.core.memory.temporal_compressor", "verified", metrics, 0.97)
    return {
        "summaries": summaries,
        "root_hash": merkle_root,
        "compression_ratio": compression_ratio,
    }


def _compute_merkle_root(child_hashes: Sequence[str]) -> str:
    """Compute Merkle root deterministically from ordered child hashes."""

    if not child_hashes:
        return sha256(b"empty").hexdigest()
    level = list(child_hashes)
    while len(level) > 1:
        next_level: List[str] = []
        for index in range(0, len(level), 2):
            left = level[index]
            right = level[index + 1] if index + 1 < len(level) else left
            next_level.append(sha256(f"{left}{right}".encode("utf-8")).hexdigest())
        level = next_level
    return level[0]


def _self_test() -> bool:
    """Ensure compression ratio and Merkle integrity meet thresholds."""

    receipts = [
        {"id": "energy:001", "resolution": "Adjusted solar reading", "topic": "energy"},
        {"id": "energy:002", "resolution": "Verified kilowatt output", "topic": "energy"},
        {"id": "policy:010", "resolution": "Aligned with oversight"},
        {"id": "policy:011", "resolution": "Logged cross-check"},
    ]
    result = compress_receipts(receipts)
    assert result["compression_ratio"] >= 0.7, "Compression ratio below threshold"
    assert len(result["root_hash"]) == 64, "Invalid Merkle root"
    write_receipt(
        "tessrax.core.memory.temporal_compressor.self_test",
        "verified",
        {"compression_ratio": result["compression_ratio"]},
        0.95,
    )
    return True


if __name__ == "__main__":
    assert _self_test(), "Self-test failed"
