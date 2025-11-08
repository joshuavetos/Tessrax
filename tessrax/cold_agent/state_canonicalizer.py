"""State canonicalization utilities for deterministic Cold Agent runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Dict

AUDIT_METADATA = {
    "auditor": "Tessrax Governance Kernel v16",
    "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
}


def _hash_state(state: Dict[str, object]) -> str:
    serialized = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


@dataclass
class CanonicalizationResult:
    pre: str
    post: str
    diff: Dict[str, object]
    state: Dict[str, object]
    previous_state: Dict[str, object]
    audit: Dict[str, object]


class StateCanonicalizer:
    """Maintains the canonical Cold Agent state and produces deterministic hashes."""

    def __init__(self) -> None:
        self.state: Dict[str, object] = {}

    def apply(self, event: Dict[str, object]) -> CanonicalizationResult:
        """Apply ``event`` to the canonical state, returning audit metadata."""

        if not isinstance(event, dict):
            raise TypeError("event must be a dictionary")

        pre_hash = _hash_state(self.state)
        previous_state = dict(self.state)
        for key, value in event.items():
            self.state[key] = value
        diff = {
            key: value
            for key, value in event.items()
            if previous_state.get(key) != value
        }
        post_hash = _hash_state(self.state)
        result = CanonicalizationResult(
            pre=pre_hash,
            post=post_hash,
            diff=diff,
            state=dict(self.state),
            previous_state=previous_state,
            audit={**AUDIT_METADATA},
        )
        assert result.pre != "" and result.post != "", "State hashes must not be empty"
        return result
