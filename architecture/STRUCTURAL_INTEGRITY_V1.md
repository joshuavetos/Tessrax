# SANITY_PROTOCOL_V1 — Phase 2 Structural Integrity Report

## Canonical Merges Executed

- Consolidated `metrics/epistemic_health.py` into the canonical `tessrax/metrics` package, removing the redundant legacy module and preserving the audited implementation.

## Merge Conflicts & Deferred Actions

- `core` → `tessrax/core`: merge blocked because v15 clients still depend on `core.ContradictionEngine`; migration remains gated on partner upgrades.
- `charts` → `docker`: Helm assets diverge from Docker Compose topology; ingress and secret mappings must be reconciled before decommissioning Helm.

## Deprecated Subsystems Scheduled for Phase 3 Deletion

- `charts`
- `core` (post-migration of downstream clients)
- `server` (empty placeholder package superseded by TIP services)

## TIP Registry Updates

- Added manifests for:
  - `tip://ledger` (`ledger/.well-known/tip.json`)
  - `tip://tests/governance-suite` (`tests/.well-known/tip.json`)
  - `tip://tessrax/core/ledger` (`tessrax/core/ledger/.well-known/tip.json`)
- Registry synchronization required no additional ledger append corrections.

## Outstanding Anomalies

- Legacy Helm templates awaiting mapping to Docker configuration.
- Legacy `core` compatibility shim pending retirement after partner migration.

## Governance Sign-Off

All changes executed under Tessrax governance clauses (AEP-001, RVC-001, EAC-001, POST-AUDIT-001). TIP manifests validated locally via `python -m tessrax.governance.tip_validate --all`. Integrity receipts recorded in the Git history for audit traceability.
