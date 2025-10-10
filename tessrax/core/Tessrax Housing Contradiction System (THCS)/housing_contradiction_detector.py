"""
Tessrax Housing Domain Detector
-------------------------------

Example domain implementation for analyzing contradictions
in housing policy and economic discourse.
"""

import hashlib
import json
from typing import Dict, Any, List
from tessrax.core.interfaces import DomainInterface


class HousingDomain(DomainInterface):
    name = "housing"

    def detect_contradictions(self, data: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Analyze housing claims for contradictions."""
        if data is None:
            data = [
                {"agent": "Developer", "claim": "Housing prices are rising."},
                {"agent": "Tenant", "claim": "Housing prices are unaffordable."},
                {"agent": "City", "claim": "We have stable housing costs."},
            ]
        contradictions = []
        for i, a in enumerate(data):
            for j, b in enumerate(data):
                if j <= i:
                    continue
                if "price" in a["claim"].lower() and "price" in b["claim"].lower():
                    contradictions.append(
                        {
                            "agents": [a["agent"], b["agent"]],
                            "reason": f"Conflicting statements about housing prices.",
                            "hash": hashlib.sha256(json.dumps([a, b]).encode()).hexdigest(),
                        }
                    )
        return {"domain": self.name, "contradictions": contradictions, "count": len(contradictions)}