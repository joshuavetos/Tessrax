"""Deterministic tests for Tessrax Merkle and anchoring subsystems."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parent


def _ensure_package(package: str, location: Path) -> None:
    if package in sys.modules:
        return
    module = types.ModuleType(package)
    module.__path__ = [str(location)]
    sys.modules[package] = module


def _load(module_name: str, relative: str):
    pkg_root = MODULE_ROOT / "tessrax"
    _ensure_package("tessrax", pkg_root)
    _ensure_package("tessrax.core", pkg_root / "core")
    target = MODULE_ROOT / relative
    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {module_name} from {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


anchor_service = _load("tessrax.core.anchor_service", "tessrax/core/anchor_service.py")
merkle_engine = _load("tessrax.core.merkle_engine", "tessrax/core/merkle_engine.py")


@pytest.fixture()
def sample_receipts() -> list[dict]:
    return [
        {"directive": "alpha", "summary": "first"},
        {"directive": "beta", "summary": "second"},
        {"directive": "gamma", "summary": "third"},
    ]


def test_build_and_proof(
    tmp_path: Path, sample_receipts: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("TESSRAX_LEDGER_PATH", str(ledger_path))
    root = merkle_engine.build_merkle_tree(sample_receipts)
    engine = merkle_engine.MerkleEngine(ledger_path)
    stored = engine.load_receipts()
    assert all("merkle_root" in receipt for receipt in stored)
    target = stored[1]
    proof = merkle_engine.generate_merkle_proof(target["receipt_id"])
    leaf_hash = merkle_engine.hash_receipt(target)
    assert merkle_engine.verify_merkle_proof(leaf_hash, proof, root)
    assert target["audit_receipt"]["integrity_score"] >= 0.9
    assert target["audit_receipt"]["signature"].startswith("SIG-")


def test_anchor_and_log(
    tmp_path: Path, sample_receipts: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    anchor_log = tmp_path / "anchors.jsonl"
    with ledger_path.open("w", encoding="utf-8") as handle:
        for item in sample_receipts:
            handle.write(json.dumps(item) + "\n")
    engine = merkle_engine.MerkleEngine(ledger_path)
    root = engine.refresh_ledger()
    monkeypatch.setenv("TESSRAX_LEDGER_PATH", str(ledger_path))
    monkeypatch.setenv("TESSRAX_ANCHOR_LOG", str(anchor_log))
    monkeypatch.setenv("ANCHOR_MODE", "mock")
    anchor_reference = anchor_service.anchor_merkle_root(root)
    updated = engine.load_receipts()
    assert any(
        r.get("external_anchor", {}).get("reference") == anchor_reference
        for r in updated
    )
    log_entries = [json.loads(line) for line in anchor_log.open("r", encoding="utf-8")]
    assert log_entries[-1]["anchor_reference"] == anchor_reference
    assert log_entries[-1]["status"] == "DLK-VERIFIED"
    assert log_entries[-1]["integrity_score"] >= 0.9


def test_self_tests() -> None:
    assert merkle_engine._self_test()
    assert anchor_service._self_test()
