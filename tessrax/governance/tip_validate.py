"""CLI utility to validate Tessrax Integration Protocol manifests.

Implements AEP-001, RVC-001, POST-AUDIT-001, and EAC-001 by performing
schema validation, signature verification, and emitting DLK receipts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema

from . import tip_registry

TIP_RECEIPT_SCHEMA = json.loads((tip_registry.SCHEMA_DIR / "tip_receipt.schema.json").read_text(encoding="utf-8"))


def _emit_receipt(receipt: dict) -> None:
    """Print a schema-validated receipt to stdout."""
    jsonschema.validate(instance=receipt, schema=TIP_RECEIPT_SCHEMA)
    sys.stdout.write(json.dumps(receipt, indent=2) + "\n")


def validate_manifest(manifest_path: Path) -> dict:
    """Load, verify, and return manifest while emitting DLK receipt."""
    try:
        manifest = tip_registry.load_tip_manifest(str(manifest_path))
        tip_registry.verify_tip_manifest(manifest)
    except Exception as exc:  # noqa: BLE001 - propagate validation issues
        receipt = tip_registry._cmd_receipt(manifest_path, success=False, error=exc)
        _emit_receipt(receipt)
        raise
    else:
        receipt = tip_registry._cmd_receipt(manifest_path, success=True, error=None)
        _emit_receipt(receipt)
        return manifest


def main(argv: list[str] | None = None) -> int:
    """Entry point for TIP manifest validation CLI."""
    parser = argparse.ArgumentParser(description="Validate TIP manifests")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all registered manifests discovered in the repository",
    )
    parser.add_argument("manifest", nargs="?", help="Path to manifest JSON file")
    args = parser.parse_args(argv)

    exit_code = 0

    if args.all:
        registry, _ = tip_registry.sync_registry()
        for entry in registry.values():
            try:
                validate_manifest(entry.path)
            except Exception:
                exit_code = 1
        return exit_code

    if not args.manifest:
        parser.error("manifest path is required unless --all is provided")

    manifest_path = Path(args.manifest)
    try:
        validate_manifest(manifest_path)
    except Exception:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
