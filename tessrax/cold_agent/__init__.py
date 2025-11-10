"""Cold Agent runtime package.

This package exposes the deterministic Cold Agent runtime components
integrated under Tessrax Governance. Each module implements an auditable
component that cooperates to validate schemas, resolve contradictions,
canonicalize state, emit receipts, append ledger entries, and audit the
Merkle root produced during execution.  All modules are stdlib-only to
comply with AEP-001 and provide runtime verification hooks per RVC-001.
"""

from .schema_validator import SchemaValidator
from .contradiction_engine import ContradictionEngine
from .state_canonicalizer import StateCanonicalizer
from .receipt_emitter import ReceiptEmitter
from .integrity_ledger import IntegrityLedger
from .audit_capsule import AuditCapsule

__all__ = [
    "SchemaValidator",
    "ContradictionEngine",
    "StateCanonicalizer",
    "ReceiptEmitter",
    "IntegrityLedger",
    "AuditCapsule",
]
