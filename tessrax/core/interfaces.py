"""
interfaces.py — Tessrax Core Interface Definitions

Defines base abstract interfaces for core modules (ILedger, IMemory, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


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