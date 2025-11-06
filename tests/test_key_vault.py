"""Unit tests for the lightweight key vault."""

from __future__ import annotations

import os
import tempfile

from tessrax.core.key_vault import KeyVault
from tessrax.core.ledger import Ledger


def test_rotation_produces_receipt() -> None:
    path = tempfile.mktemp()
    ledger = Ledger(path)
    vault = KeyVault(ledger)
    receipt = vault.rotate_key()
    assert "chain_hash" in receipt
    assert os.path.exists(vault.current_key_file)
