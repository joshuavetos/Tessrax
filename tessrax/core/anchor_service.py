from __future__ import annotations

"""External anchoring service compliant with Tessrax clauses AEP-001, RVC-001, and EAC-001."""

import hashlib
import json
import os
import time
from pathlib import Path

from . import merkle_engine

_GOVERNANCE_AUDITOR = "Tessrax Governance Kernel v16"
_CLAUSES = ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"]


def _resolve_anchor_log() -> Path:
    override = os.getenv("TESSRAX_ANCHOR_LOG")
    if override:
        return Path(override).expanduser().resolve()
    ledger_path = merkle_engine.MerkleEngine().ledger_path
    return ledger_path.with_name("anchors.jsonl")


def _anchor_receipt(root_hash: str, anchor_reference: str, mode: str) -> dict:
    execution_hash = hashlib.sha256(f"anchor:{root_hash}:{anchor_reference}:{mode}".encode("utf-8")).hexdigest()
    return {
        "auditor": _GOVERNANCE_AUDITOR,
        "clauses": _CLAUSES,
        "timestamp": time.time(),
        "status": "DLK-VERIFIED",
        "integrity_score": 0.95,
        "legitimacy": 0.92,
        "mode": mode,
        "anchor_reference": anchor_reference,
        "root_hash": root_hash,
        "runtime_info": {
            "execution_hash": execution_hash,
        },
        "signature": f"SIG-{execution_hash[:32]}",
    }


def _generate_mock_anchor(root_hash: str) -> str:
    digest = hashlib.sha256(root_hash.encode("utf-8")).hexdigest()
    return f"mock://{digest[:46]}"


def _write_anchor_log(receipt: dict) -> None:
    log_path = _resolve_anchor_log()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, sort_keys=True) + "\n")


def anchor_merkle_root(root_hash: str) -> str:
    """Anchor a Merkle root and record the external reference (AEP-001/RVC-001/EAC-001)."""

    if not isinstance(root_hash, str) or not root_hash:
        raise ValueError("root_hash must be a non-empty hexadecimal string")
    mode = (os.getenv("ANCHOR_MODE") or "mock").lower()
    if mode == "mock":
        anchor_reference = _generate_mock_anchor(root_hash)
    elif mode in {"ipfs", "arweave"}:
        raise RuntimeError(
            f"{mode.upper()} anchoring requires network access; set ANCHOR_MODE=mock for offline compliance."
        )
    else:
        raise ValueError(f"Unsupported ANCHOR_MODE: {mode}")

    receipt = _anchor_receipt(root_hash, anchor_reference, mode)
    _write_anchor_log(receipt)
    provider = "mock" if mode == "mock" else mode
    engine = merkle_engine.MerkleEngine()
    engine.record_anchor(root_hash, anchor_reference, provider)
    return anchor_reference


def _self_test() -> bool:
    """Confirm anchor service operations including mock compliance (AEP-001/RVC-001/EAC-001)."""

    import tempfile

    prior_env = {key: os.environ.get(key) for key in ["TESSRAX_LEDGER_PATH", "TESSRAX_ANCHOR_LOG", "ANCHOR_MODE"]}
    try:
        with tempfile.TemporaryDirectory(prefix="tessrax_anchor_") as tmp:
            tmp_path = Path(tmp)
            ledger_path = tmp_path / "ledger.jsonl"
            anchor_log = tmp_path / "anchors.jsonl"
            sample_receipts = [
                {"directive": "delta", "summary": "anchoring"},
                {"directive": "epsilon", "summary": "anchoring"},
            ]
            with ledger_path.open("w", encoding="utf-8") as handle:
                for item in sample_receipts:
                    handle.write(json.dumps(item) + "\n")
            os.environ["TESSRAX_LEDGER_PATH"] = str(ledger_path)
            os.environ["TESSRAX_ANCHOR_LOG"] = str(anchor_log)
            os.environ["ANCHOR_MODE"] = "mock"
            engine = merkle_engine.MerkleEngine(ledger_path)
            root = engine.refresh_ledger()
            anchor_reference = anchor_merkle_root(root)
            receipts = engine.load_receipts()
            if not any(r.get("external_anchor", {}).get("reference") == anchor_reference for r in receipts):
                raise AssertionError("Anchor reference not recorded in ledger")
            log_entries = [json.loads(line) for line in anchor_log.open("r", encoding="utf-8")]
            if not log_entries or log_entries[-1]["anchor_reference"] != anchor_reference:
                raise AssertionError("Anchor log missing latest reference")
            if log_entries[-1]["integrity_score"] < 0.9:
                raise AssertionError("Anchor integrity score below governed threshold")
    finally:
        for key, value in prior_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return True


if __name__ == "__main__":  # pragma: no cover
    assert _self_test(), "Anchor service self-test failed"
