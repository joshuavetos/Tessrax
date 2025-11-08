"""Tessrax core package marker enforcing governance integrity."""

try:
    import sys
    import types
    import nacl.encoding
    import nacl.signing

    ed25519 = types.SimpleNamespace(
        SigningKey=nacl.signing.SigningKey,
        VerifyKey=nacl.signing.VerifyKey,
        encoding=nacl.encoding,
    )
    sys.modules.setdefault("ed25519", ed25519)
except Exception as exc:  # pragma: no cover - defensive guard
    print("⚠️ ed25519 alias patch failed:", exc)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = (PROJECT_ROOT / "ledger" / "ledger.jsonl").resolve()
LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
if not LEDGER_PATH.exists():
    LEDGER_PATH.write_text("", encoding="utf-8")

from . import audit_kernel  # noqa: F401  # DLK-VERIFIED lazy adapter

from . import ledger  # noqa: F401  # DLK-VERIFIED import exposure

__all__ = ["PROJECT_ROOT", "LEDGER_PATH", "audit_kernel"]
