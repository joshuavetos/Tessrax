"""Integration tests for the Tessrax Integration Protocol implementation."""
from __future__ import annotations

import json
import jsonschema
from pathlib import Path

import pytest

from tessrax.governance import tip_registry, tip_validate


@pytest.fixture()
def manifest_template() -> dict:
    """Return a base manifest dictionary without the signature hash."""
    return {
        "module_id": "example@sha256:abcd",
        "version": "1.2.3",
        "description": "Example TIP compliant service",
        "endpoints": [
            {
                "name": "status",
                "method": "GET",
                "route": "/status",
                "io_schema_hash": "a" * 64
            }
        ],
        "cli_entrypoints": ["python -m tessrax.example"],
        "dependencies": ["tip://ledger-core#abc"],
        "health": {
            "readiness": "https://localhost/readyz",
            "liveness": "https://localhost/healthz",
            "last_checked": "2025-02-18T00:00:00Z"
        },
        "telemetry_topics": ["governance.signals.drift"],
        "governance": {
            "signing_key_id": "ledger-core-ed25519-01",
            "signature": {
                "algorithm": "SHA256",
                "value": "",
                "signed_at": "2025-02-18T00:00:00Z"
            }
        },
        "governance_bus": {
            "ingest": "/governance/receipts",
            "emit": "/governance/signals/{id}",
            "status_broadcast": "/governance/status"
        },
        "ledger_refs": [
            {
                "path": "/ledger/example.jsonl",
                "hash": "b" * 64,
                "merkle_anchor": "bafyexample"
            }
        ],
        "federation_profile": {
            "integrity_score": 0.95,
            "reliability": {
                "uptime_24h": 0.999,
                "error_budget_consumed": 0.01
            },
            "compliance": {
                "aep_001": True,
                "rvc_001": True,
                "post_audit_001": True,
                "eac_001": True
            },
            "scar_register": [],
            "status": "ready",
            "last_self_audit": "2025-02-18T00:00:00Z"
        }
    }


def _write_manifest(path: Path, manifest: dict) -> dict:
    manifest_copy = json.loads(json.dumps(manifest))
    manifest_copy["governance"]["signature"]["value"] = tip_registry._hash_manifest(manifest_copy)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest_copy, indent=2), encoding="utf-8")
    return manifest_copy


def test_manifest_validation_and_registration(tmp_path: Path, manifest_template: dict) -> None:
    manifest_path = tmp_path / ".well-known" / "tip.json"
    manifest = _write_manifest(manifest_path, manifest_template)

    loaded = tip_registry.load_tip_manifest(str(manifest_path))
    assert loaded["module_id"] == manifest["module_id"]

    assert tip_registry.verify_tip_manifest(loaded) is True

    ledger_file = tmp_path / "tip_manifests.jsonl"
    record = tip_registry.register_module(loaded, ledger_path=ledger_file)
    assert record["module_id"] == manifest["module_id"]
    assert ledger_file.is_file()
    lines = ledger_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    stored_record = json.loads(lines[0])
    assert stored_record["manifest_hash"] == tip_registry._hash_manifest(manifest)
    assert stored_record["ready_for_federation"] is True
    assert stored_record["scar_register"] == []


def test_manifest_with_low_integrity_rejected(tmp_path: Path, manifest_template: dict) -> None:
    manifest_path = tmp_path / "tip.json"
    manifest_template["federation_profile"]["integrity_score"] = 0.5
    manifest = _write_manifest(manifest_path, manifest_template)

    loaded = tip_registry.load_tip_manifest(str(manifest_path))
    with pytest.raises(ValueError, match="Integrity score below federation threshold"):
        tip_registry.verify_tip_manifest(loaded)


def test_discover_reports_invalid_and_valid(tmp_path: Path, manifest_template: dict) -> None:
    valid_path = tmp_path / "service1" / ".well-known" / "tip.json"
    invalid_path = tmp_path / "service2" / ".well-known" / "tip.json"

    _write_manifest(valid_path, manifest_template)

    invalid_manifest = manifest_template.copy()
    invalid_manifest["federation_profile"] = invalid_manifest["federation_profile"].copy()
    invalid_manifest["federation_profile"]["integrity_score"] = 0.2
    _write_manifest(invalid_path, invalid_manifest)

    results = tip_registry.discover_manifests(tmp_path)
    assert len(results) == 2
    ready = [entry for entry in results if entry.get("ready_for_federation")]
    invalid = [entry for entry in results if entry["status"].startswith("invalid")]
    assert len(ready) == 1
    assert len(invalid) == 1


def test_discover_handles_invalid_json(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "broken" / ".well-known"
    manifest_dir.mkdir(parents=True)
    manifest_path = manifest_dir / "tip.json"
    manifest_path.write_text("not-json", encoding="utf-8")

    results = tip_registry.discover_manifests(tmp_path)
    assert len(results) == 1
    entry = results[0]
    assert entry["module_id"].endswith("tip.json")
    assert entry["ready_for_federation"] is False
    assert entry["status"].startswith("invalid")


def test_tip_validate_cli_emits_receipt(tmp_path: Path, manifest_template: dict, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = tmp_path / "tip.json"
    manifest = _write_manifest(manifest_path, manifest_template)

    exit_code = tip_validate.main([str(manifest_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    receipt = json.loads(out)
    jsonschema.validate(receipt, tip_registry.TIP_RECEIPT_SCHEMA)
    assert receipt["status"] == "DLK-VERIFIED"

    manifest_bad = json.loads(json.dumps(manifest))
    manifest_bad["federation_profile"]["integrity_score"] = 0.1
    bad_path = tmp_path / "bad.json"
    bad_manifest = _write_manifest(bad_path, manifest_bad)
    bad_manifest["governance"]["signature"]["value"] = tip_registry._hash_manifest(bad_manifest)
    bad_path.write_text(json.dumps(bad_manifest, indent=2), encoding="utf-8")

    exit_code_bad = tip_validate.main([str(bad_path)])
    assert exit_code_bad == 1
    out_bad = capsys.readouterr().out
    failed_receipt = json.loads(out_bad)
    jsonschema.validate(failed_receipt, tip_registry.TIP_RECEIPT_SCHEMA)
    assert failed_receipt["status"] == "DLK-FAILED"


def test_schema_validation_catches_missing_fields(tmp_path: Path, manifest_template: dict) -> None:
    manifest_path = tmp_path / "tip.json"
    broken = manifest_template.copy()
    broken.pop("module_id")
    manifest_path.write_text(json.dumps(broken, indent=2), encoding="utf-8")

    with pytest.raises(jsonschema.ValidationError):
        tip_registry.load_tip_manifest(str(manifest_path))
