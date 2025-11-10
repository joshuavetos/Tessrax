"""
Simple ledger self-test ensuring importability and verify() callable.
"""
from __future__ import annotations

import sys

from core import verify_ledger


def self_test() -> bool:
    try:
        result = verify_ledger.verify()
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        print("Ledger self_test raised:", exc)
        return False
    else:
        print("✅ Ledger self_test passed" if result else "❌ Ledger self_test failed")
        return bool(result)


if __name__ == "__main__":
    if not self_test():
        sys.exit(1)
