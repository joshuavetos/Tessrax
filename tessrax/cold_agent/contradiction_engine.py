"""Contradiction detection logic for the Cold Agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Dict, List

AUDIT_METADATA = {
    "auditor": "Tessrax Governance Kernel v16",
    "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
}


@dataclass(frozen=True)
class ContradictionRecord:
    """Container describing a contradiction between historical and new state."""

    key: str
    old: object
    new: object


@dataclass(frozen=True)
class ContradictionResult:
    """Deterministic result emitted after contradiction analysis."""

    contradictions: List[ContradictionRecord]
    post_hash: str
    audit: Dict[str, object]


def _hash_state(state: Dict[str, object]) -> str:
    serialized = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class ContradictionEngine:
    """Detects conflicting updates in the canonical Cold Agent state."""

    def __init__(self) -> None:
        self._last_state_hash: str | None = None

    def detect(
        self,
        previous_state: Dict[str, object],
        event: Dict[str, object],
        resulting_state: Dict[str, object] | None = None,
    ) -> ContradictionResult:
        """Return contradictions between ``previous_state`` and ``event``.

        ``resulting_state`` may be provided when the canonical state was
        already updated.  The returned ``post_hash`` will reference the
        resulting state when available; otherwise it represents the
        ``previous_state`` hash.  This keeps ledger computation
        deterministic regardless of invocation order.
        """

        if not isinstance(previous_state, dict) or not isinstance(event, dict):
            raise TypeError("previous_state and event must be dictionaries")
        if resulting_state is not None and not isinstance(resulting_state, dict):
            raise TypeError("resulting_state must be a dictionary when provided")

        contradictions: List[ContradictionRecord] = []
        for key, value in event.items():
            if key in previous_state and previous_state[key] != value:
                contradictions.append(ContradictionRecord(key=key, old=previous_state[key], new=value))

        target_state = resulting_state if resulting_state is not None else previous_state
        post_hash = _hash_state(target_state)
        self._last_state_hash = post_hash
        return ContradictionResult(
            contradictions=contradictions,
            post_hash=post_hash,
            audit={**AUDIT_METADATA},
        )

    @property
    def last_state_hash(self) -> str | None:
        """Expose the last computed state hash for auditability."""

        return self._last_state_hash
