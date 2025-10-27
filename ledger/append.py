"""Hybrid ledger append strategy."""
from __future__ import annotations

import os

from ledger.merkle import append_merkle
from ledger.local import append_local


def append(entry):
    """Append an entry using the environment-aware backend."""
    if os.getenv("TESSRAX_ENV") == "prod":
        return append_merkle(entry)
    return append_local(entry)
