"""
Tessrax Core Interfaces
-----------------------

Defines the abstract interface that all domain-specific contradiction
detectors must implement. This ensures cross-domain consistency and
plug-and-play extensibility.
"""

from typing import Any, Dict

class DomainInterface:
    """Base class for all Tessrax domain modules."""

    name: str = "generic"

    def detect_contradictions(self, data: Any = None) -> Dict[str, Any]:
        """
        Run contradiction detection logic on provided input data.
        Must return a serializable dict or list representing results.
        """
        raise NotImplementedError("Subclasses must implement detect_contradictions().")

    def to_json(self, result: Any) -> Dict[str, Any]:
        """Default JSON serialization hook."""
        if isinstance(result, dict):
            return result
        return {"result": str(result)}