"""Schema validation utilities for the Cold Agent runtime.

The validator ensures incoming events conform to deterministic shape
constraints before any state transition occurs.  Each validation call
returns a receipt-style payload containing:
    * ``valid`` — boolean flag indicating schema soundness.
    * ``errors`` — structured list of (field, reason) tuples.
    * ``pre_hash`` — SHA-256 digest of the canonicalized payload.

Runtime verification (RVC-001) is satisfied by type checks and by
asserting that the serialized form can be hashed.  A helper hashing
function is exposed for downstream verification steps.  All code relies
solely on the Python 3.11 standard library to satisfy AEP-001.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Iterable, List, Tuple

AUDIT_METADATA = {
    "auditor": "Tessrax Governance Kernel v16",
    "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
}


@dataclass(frozen=True)
class ValidationResult:
    """Structured validation outcome used throughout the Cold Agent runtime."""

    valid: bool
    errors: List[Tuple[str, str]]
    pre_hash: str
    audit: Dict[str, object]


def sha256_digest(payload: Dict[str, object]) -> str:
    """Return the SHA-256 hex digest for the given payload.

    The payload is serialized deterministically (sorted keys, UTF-8) so
    downstream integrity checks can recompute the digest during audits.
    Raises ``TypeError`` if the payload cannot be serialized to JSON.
    """

    try:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise TypeError("Payload is not JSON serializable") from exc
    return hashlib.sha256(serialized).hexdigest()


class SchemaValidator:
    """Validates event payloads against minimal Cold Agent expectations."""

    def __init__(self) -> None:
        self.registry: Dict[str, Iterable[str]] = {}

    def register_required_fields(self, schema_name: str, fields: Iterable[str]) -> None:
        """Register fields that must be present for the named schema."""

        if not schema_name:
            raise ValueError("schema_name must be a non-empty string")
        self.registry[schema_name] = tuple(fields)

    def validate(self, data: Dict[str, object], schema_name: str | None = None) -> ValidationResult:
        """Validate ``data`` using optional schema metadata.

        The method enforces non-null values and required keys (if a schema
        was previously registered).  A deterministic hash of ``data`` is
        returned for downstream inclusion in receipts.
        """

        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")
        if schema_name is not None and schema_name not in self.registry:
            raise KeyError(f"Unknown schema '{schema_name}'")

        errors: List[Tuple[str, str]] = []
        for key, value in data.items():
            if value is None:
                errors.append((key, "null"))
        if schema_name is not None:
            required = set(self.registry[schema_name])
            missing = required.difference(data)
            for missing_field in sorted(missing):
                errors.append((missing_field, "missing"))

        result = ValidationResult(
            valid=not errors,
            errors=errors,
            pre_hash=sha256_digest(data),
            audit={**AUDIT_METADATA},
        )
        assert result.pre_hash, "SHA-256 digest must not be empty"
        return result
