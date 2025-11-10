"""Tessrax Integration Protocol registry utilities.

This module enforces AEP-001, RVC-001, POST-AUDIT-001, and EAC-001 by
validating manifests, verifying signatures, and maintaining the
append-only ledger of registered modules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import jsonschema

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
TIP_MANIFEST_SCHEMA = json.loads((SCHEMA_DIR / "tip_manifest.schema.json").read_text(encoding="utf-8"))
TIP_RECEIPT_SCHEMA = json.loads((SCHEMA_DIR / "tip_receipt.schema.json").read_text(encoding="utf-8"))

LEDGER_PATH = Path(__file__).resolve().parents[1] / "ledger" / "tip_manifests.jsonl"


def _canonical_json(data: Dict) -> str:
    """Return canonical JSON string for hashing with stable key ordering."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def load_tip_manifest(path: str) -> Dict:
    """Load and schema-validate a TIP manifest.

    Raises:
        FileNotFoundError: if the manifest file does not exist.
        json.JSONDecodeError: if the file is not valid JSON.
        jsonschema.ValidationError: if the manifest violates the schema.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=data, schema=TIP_MANIFEST_SCHEMA)
    return data


def _hash_manifest(manifest: Dict) -> str:
    """Compute SHA-256 hash of the manifest without the signature value."""
    manifest_copy = json.loads(json.dumps(manifest))
    signature = manifest_copy.get("governance", {}).get("signature", {})
    if "value" in signature:
        signature.pop("value")
    digest = hashlib.sha256(_canonical_json(manifest_copy).encode("utf-8")).hexdigest()
    return digest


def verify_tip_manifest(manifest: Dict) -> bool:
    """Verify manifest signature and compliance clauses.

    Returns:
        bool: True if verification succeeds.

    Raises:
        ValueError: if verification fails.
    """
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a dictionary")

    governance = manifest.get("governance")
    if not governance:
        raise ValueError("Manifest missing governance section")
    signature = governance.get("signature")
    if not signature:
        raise ValueError("Manifest missing governance.signature")
    if signature.get("algorithm") != "SHA256":
        raise ValueError("Unsupported signature algorithm; expected SHA256")

    expected_hash = _hash_manifest(manifest)
    provided_hash = signature.get("value")
    if expected_hash != provided_hash:
        raise ValueError("Manifest signature hash mismatch")

    federation = manifest.get("federation_profile")
    if not federation:
        raise ValueError("Manifest missing federation_profile")

    integrity = federation.get("integrity_score")
    if integrity is None:
        raise ValueError("Manifest missing integrity_score")
    if not isinstance(integrity, (int, float)):
        raise ValueError("Integrity score must be numeric")
    if not 0 <= integrity <= 1:
        raise ValueError("Integrity score must be within [0, 1]")

    compliance = federation.get("compliance")
    if not compliance:
        raise ValueError("Manifest missing compliance section")
    missing = [clause for clause in ("aep_001", "rvc_001", "post_audit_001", "eac_001") if not compliance.get(clause)]
    if missing:
        raise ValueError(f"Manifest violates clauses: {', '.join(missing)}")

    if integrity < 0.9:
        raise ValueError("Integrity score below federation threshold of 0.9")

    return True


def register_module(manifest: Dict, ledger_path: Path | None = None) -> Dict:
    """Append a manifest registration record to the ledger.

    Args:
        manifest: The verified manifest dictionary.
        ledger_path: Optional override for ledger path (used in tests).

    Returns:
        The ledger record appended.
    """
    if ledger_path is None:
        ledger_path = LEDGER_PATH

    verify_tip_manifest(manifest)

    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    scar_register = manifest["federation_profile"].get("scar_register", [])
    ready_for_federation = (
        manifest["federation_profile"]["integrity_score"] >= 0.9
        and not scar_register
        and manifest["federation_profile"].get("status") == "ready"
    )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module_id": manifest["module_id"],
        "version": manifest["version"],
        "integrity_score": manifest["federation_profile"]["integrity_score"],
        "status": manifest["federation_profile"].get("status", "unknown"),
        "manifest_hash": _hash_manifest(manifest),
        "ready_for_federation": ready_for_federation,
        "scar_register": scar_register,
    }

    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(record) + "\n")

    return record


def _discover_manifests(directory: Path) -> Iterable[Path]:
    """Yield manifest paths from .well-known/tip.json under directory."""
    for manifest_path in directory.rglob(".well-known/tip.json"):
        if manifest_path.is_file():
            yield manifest_path


def _build_readiness_entry(manifest: Dict) -> Dict:
    """Construct readiness information for a manifest."""
    federation = manifest["federation_profile"]
    scar_register = federation.get("scar_register", [])
    integrity = federation["integrity_score"]
    ready = integrity >= 0.9 and not scar_register and federation.get("status") == "ready"
    return {
        "module_id": manifest["module_id"],
        "integrity_score": integrity,
        "status": federation.get("status", "unknown"),
        "scar_register": scar_register,
        "ready_for_federation": ready,
    }


def discover_manifests(directory: Path) -> List[Dict]:
    """Locate, validate, and summarise manifests inside directory."""
    readiness: List[Dict] = []
    for manifest_path in _discover_manifests(directory):
        try:
            manifest = load_tip_manifest(str(manifest_path))
        except (json.JSONDecodeError, FileNotFoundError, jsonschema.ValidationError) as exc:
            readiness.append({
                "module_id": str(manifest_path),
                "integrity_score": None,
                "status": f"invalid: {exc}",
                "scar_register": [],
                "ready_for_federation": False,
            })
            continue

        try:
            verify_tip_manifest(manifest)
            readiness.append(_build_readiness_entry(manifest))
        except ValueError as exc:
            federation = manifest.get("federation_profile", {})
            readiness.append({
                "module_id": manifest.get("module_id", str(manifest_path)),
                "integrity_score": federation.get("integrity_score"),
                "status": f"invalid: {exc}",
                "scar_register": federation.get("scar_register", []),
                "ready_for_federation": False,
            })
    return readiness


def _print_readiness_map(entries: List[Dict]) -> None:
    """Print readiness map as JSON array."""
    sys.stdout.write(json.dumps(entries, indent=2) + "\n")


def _cmd_discover(args: argparse.Namespace) -> int:
    directory = Path(args.dir).resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    entries = discover_manifests(directory)
    _print_readiness_map(entries)
    return 0


def _build_receipt(status: str, runtime_info: str, details: Dict | None = None, integrity_score: float = 1.0) -> Dict:
    """Create a DLK receipt ensuring compliance with POST-AUDIT-001."""
    payload = {
        "auditor": "Tessrax Governance Kernel v16",
        "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_info": runtime_info,
        "integrity_score": integrity_score,
        "status": status,
        "signature": hashlib.sha256(runtime_info.encode("utf-8")).hexdigest()
    }
    if details:
        payload["details"] = details
    jsonschema.validate(instance=payload, schema=TIP_RECEIPT_SCHEMA)
    return payload


def _cmd_receipt(manifest_path: Path, success: bool, error: Exception | None) -> Dict:
    if success:
        runtime_info = f"Validated manifest at {manifest_path}"
        return _build_receipt("DLK-VERIFIED", runtime_info, {"manifest": str(manifest_path)}, 0.96)
    runtime_info = f"Validation failed for {manifest_path}: {error}"
    return _build_receipt("DLK-FAILED", runtime_info, {"manifest": str(manifest_path)}, 0.0)


def main(argv: List[str] | None = None) -> int:
    """CLI entry point for manifest discovery and ledger operations."""
    parser = argparse.ArgumentParser(description="Tessrax TIP registry tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser("discover", help="Discover manifests in directory")
    discover_parser.add_argument("--dir", default=".", help="Directory to scan for manifests")

    args = parser.parse_args(argv)

    if args.command == "discover":
        return _cmd_discover(args)

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
