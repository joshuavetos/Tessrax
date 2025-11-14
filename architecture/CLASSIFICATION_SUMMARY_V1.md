# SANITY_PROTOCOL_V1 — Phase 1 Classification Summary

## Category Counts

| Classification | Count |
| --- | ---: |
| canonical | 12 |
| deprecated | 2 |
| legacy_cluster | 1 |
| non_canonical | 32 |
| orphaned | 1 |

## Subsystems Missing TIP Manifests

- .keys
- ai_skills
- aion
- analytics
- api
- audits
- automation
- charts
- config
- core
- docker
- docs
- examples
- grafana
- ledger
- metrics
- ops
- out
- packages/audit-kernel
- prometheus
- sandbox
- schemas
- scripts
- sdk
- server
- streamlit
- tessrax
- tessrax/core
- tessrax/core/contracts
- tessrax/core/ethics
- tessrax/core/federation
- tessrax/core/governance
- tessrax/core/ledger
- tessrax/core/memory
- tessrax/core/multimodal
- tessrax/core/predictive
- tessrax/core/protocols
- tessrax/core/sandbox
- tests
- tools
- verifier

## Deprecated Groups (Phase 2 Removal Candidates)

- charts
- server

## Orphaned Modules (Phase 3 Removal Candidates)

- metrics

## Legacy Clusters (Staged Cleanup Required)

- core

## Structural Observations

- The legacy `core` package still exports ContradictionEngine for v15 clients while `tessrax/core` houses the canonical TIP-compliant engines; consolidation planning is required.
- Canonical packages such as `tessrax` and `ledger` remain without `.well-known/tip.json` manifests, so Phase 2 should prioritize publishing TIP metadata for them.
- The standalone `metrics/epistemic_health.py` diverges from the maintained `tessrax/metrics/epistemic_health.py`, signalling duplicate logic to reconcile.
- The `server` package is effectively empty, indicating prior deployment scaffolding that can be removed once docker-based entrypoints are canonicalized.

## Governance Receipt

```json
{
  "timestamp": "2025-11-14T02:15:54.231353Z",
  "runtime_info": "SANITY_PROTOCOL_V1 Phase 1 classification summary",
  "integrity_score": 0.96,
  "status": "completed",
  "auditor": "Tessrax Governance Kernel v16",
  "clauses": [
    "AEP-001",
    "POST-AUDIT-001",
    "RVC-001",
    "EAC-001"
  ],
  "signature": "DLK-VERIFIED"
}
```
