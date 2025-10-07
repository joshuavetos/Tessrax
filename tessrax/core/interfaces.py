"""
interfaces.py — Tessrax Core Interface Definitions

Defines base abstract interfaces for core modules (ILedger, IMemory, IReceipt).
These interfaces establish formal contracts for interchangeable implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


# ============================================================
# Ledger Interface
# ============================================================
class ILedger(ABC):
    """Interface contract for all Tessrax ledger implementations."""

    @abstractmethod
    def add_event(self, event: Dict[str, Any]) -> str:
        """Append an event and return its hash."""
        pass

    @abstractmethod
    def get_all_events(self, verify: bool = True) -> List[Dict[str, Any]]:
        """Return all events, optionally verifying the chain."""
        pass

    @abstractmethod
    def verify_chain(self) -> bool:
        """Verify chain integrity."""
        pass

    @abstractmethod
    def merkle_root(self) -> Optional[str]:
        """Return the current Merkle root hash."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close any resources or connections."""
        pass


# ============================================================
# Memory Interface
# ============================================================
class IMemory(ABC):
    """Interface for contradiction-aware, provenance-tracking memory systems."""

    @abstractmethod
    def add(self, key: str, value: Any, provenance: str) -> None:
        """Add a key/value entry with provenance tracking."""
        pass

    @abstractmethod
    def get(self, key: str) -> Dict[str, Any]:
        """Retrieve a memory entry."""
        pass

    @abstractmethod
    def export(self) -> str:
        """Export memory contents as JSON."""
        pass

    @abstractmethod
    def all_keys(self) -> List[str]:
        """Return list of all stored keys."""
        pass


# ============================================================
# Receipt Interface
# ============================================================
class IReceipt(ABC):
    """Interface defining creation and verification of cryptographic receipts."""

    @abstractmethod
    def create_receipt(self, private_key_hex: str, event_payload: Dict[str, Any], executor_id: str = "unknown") -> Dict[str, Any]:
        """Create a signed receipt from an event payload."""
        pass

    @abstractmethod
    def verify_receipt(self, receipt: Dict[str, Any]) -> bool:
        """Verify authenticity and integrity of a receipt."""
        pass