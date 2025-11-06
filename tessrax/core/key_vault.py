"""Tessrax Key-Vault & Rotation Engine (v18.1).

Provides deterministic Ed25519 key rotation anchored to the lightweight
ledger defined in :mod:`tessrax.core.ledger`.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

from nacl.signing import SigningKey

from tessrax.core.ledger import Ledger


class KeyVault:
    """Manage signing keys and record rotations in the ledger."""

    def __init__(self, ledger: Ledger, path: str | os.PathLike[str] = ".keys") -> None:
        self.ledger = ledger
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.current_key_file = self.path / "current.key"
        if not self.current_key_file.exists():
            self._generate_initial_key()

    # ------------------------------------------------------------------
    # Key generation utilities
    # ------------------------------------------------------------------
    def _generate_initial_key(self) -> None:
        signing_key = SigningKey.generate()
        self._write_key(signing_key)
        self._log_event("INITIAL_KEY_CREATED", signing_key)

    def _write_key(self, signing_key: SigningKey) -> None:
        with self.current_key_file.open("wb") as handle:
            handle.write(signing_key.encode())

    def _log_event(self, event: str, signing_key: SigningKey) -> None:
        payload = {
            "directive": f"KEY_EVENT_{event}",
            "timestamp": self._timestamp(),
            "public_key": signing_key.verify_key.encode().hex(),
            "key_hash": hashlib.sha256(signing_key.encode()).hexdigest(),
        }
        self.ledger.append(payload)

    @staticmethod
    def _timestamp() -> str:
        return _dt.datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def rotate_key(self) -> Mapping[str, Any]:
        """Rotate the signing key and append a rotation receipt to the ledger."""

        previous_material = self.current_key_file.read_bytes()
        new_key = SigningKey.generate()
        self._write_key(new_key)
        rotation_payload = {
            "directive": "KEY_ROTATION",
            "timestamp": self._timestamp(),
            "old_hash": hashlib.sha256(previous_material).hexdigest(),
            "new_hash": hashlib.sha256(new_key.encode()).hexdigest(),
            "chain_hash": hashlib.sha256(previous_material + new_key.encode()).hexdigest(),
        }
        self.ledger.append(rotation_payload)
        return rotation_payload

    def current_verify_key(self) -> bytes:
        """Return the public verify key associated with the stored signing key."""

        signing_key = SigningKey(self.current_key_file.read_bytes())
        return signing_key.verify_key.encode()


__all__ = ["KeyVault"]
