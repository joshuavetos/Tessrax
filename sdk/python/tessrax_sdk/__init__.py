"""
Tessrax Python SDK — v19.6
Provides programmatic access to emit, verify, and query governance receipts.
"""

import json
import requests
from typing import Any, Dict, Optional

BASE_URL = "http://localhost:8080"

def emit_receipt(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Emit a governance receipt to the Tessrax node."""
    r = requests.post(f"{BASE_URL}/api/receipts", json=payload, timeout=5)
    r.raise_for_status()
    return r.json()

def verify_receipt(receipt: Dict[str, Any]) -> bool:
    """Verify a receipt’s Merkle inclusion proof via the node."""
    r = requests.post(f"{BASE_URL}/api/verify", json=receipt, timeout=5)
    r.raise_for_status()
    return r.json().get("verified", False)

def status(window: int = 100) -> Dict[str, Any]:
    """Return system metrics and recent receipts."""
    r = requests.get(f"{BASE_URL}/api/status?window={window}", timeout=5)
    r.raise_for_status()
    return r.json()

__all__ = ["emit_receipt", "verify_receipt", "status"]
