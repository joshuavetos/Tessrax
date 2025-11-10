"""Compatibility shim re-exporting Tessrax receipt utilities."""
from __future__ import annotations

from tessrax.core.governance.receipts import Receipt, write_receipt

__all__ = ["Receipt", "write_receipt"]
