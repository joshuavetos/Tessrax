"""Tessrax core package marker enforcing governance integrity."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = (PROJECT_ROOT / "ledger" / "ledger.jsonl").resolve()
LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
if not LEDGER_PATH.exists():
    LEDGER_PATH.write_text("", encoding="utf-8")

__all__ = ["PROJECT_ROOT", "LEDGER_PATH"]
