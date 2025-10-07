"""
interfaces.py — Tessrax Core Interface Definitions

Defines abstract base interfaces for core Tessrax modules:
- ILedger: Append-only, verifiable event store
- IMemory: Provenance-tracking memory system
- IReceipt: Cryptographic receipt generator/verifier
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


# ============================================================
# ILedger — Event Ledger Interface
# ============================================================

class ILedger(ABC):
    """
    Abstract interface for append-only, tamper-evident event ledgers.
    Implementations must ensure integrity, verification, and auditability.
    """

    @abstractmethod
    def add_event(self, event: Dict[str, Any]) -> str:
        """Append a single event to the ledger and return its hash."""
        pass

    @abstractmethod
    def get_all_events(self, verify: bool = True) -> List[Dict[str, Any]]:
        """Retrieve all events in ledger order. Optionally verify hash-chain integrity."""
        pass

    @abstractmethod
    def verify_chain(self) -> bool:
        """Verify continuity and validity of the event hash chain."""
        pass

    @abstractmethod
    def merkle_root(self) -> Optional[str]:
        """Return the Merkle root of all current entries, or None if empty."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close any persistent resources (e.g., database handles)."""
        pass


# ============================================================
# IMemory — Provenance-Aware Memory Interface
# ============================================================

class IMemory(ABC):
    """
    Interface for contradiction-aware, provenance-tracking memory modules.
    Supports key-value storage with embedded origin metadata.
    """

    @abstractmethod
    def add(self, key: str, value: Any, provenance: str) -> None:
        """Store a key-value pair along with its provenance (e.g., source agent)."""
        pass

    @abstractmethod
    def get(self, key: str) -> Dict[str, Any]:
        """Retrieve the full memory entry, including provenance."""
        pass

    @abstractmethod
    def export(self) -> str:
        """Export the full memory contents as a JSON string."""
        pass

    @abstractmethod
    def all_keys(self) -> List[str]:
        """Return a list of all keys currently stored in memory."""
        pass


# ============================================================
# IReceipt — Cryptographic Receipt Interface
# ============================================================

class IReceipt(ABC):
    """
    Interface for generating and verifying tamper-evident receipts.
    Receipts are signed, timestamped proofs of structured actions or claims.
    """

    @abstractmethod
    def create_receipt(
        self,
        private_key_hex: str,
        event_payload: Dict[str, Any],
        executor_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Create a cryptographically signed receipt for an event payload.
        Returns the full receipt object (with hashes, signature, etc.).
        """
        pass

    @abstractmethod
    def verify_receipt(self, receipt: Dict[str, Any]) -> bool:
        """
        Verify the authenticity, integrity, and signature of a receipt.
        Returns True if valid, False or exception if tampered.
        """
        pass