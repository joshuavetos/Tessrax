# Tessrax Contradiction Curriculum (DLK-Verified)

This curriculum extends the AI Skills Lab with governed Tessrax capabilities.
Each module references real repository assets and encourages reflections via the
Human Feedback API (`POST /feedback/`).  Learners can explore prompts via the
ethical sandbox in `sandbox/ethical_drift.py`.

## Modules

1. **Governance Foundations** — Review `docs/architecture_overview.md` and map
the Memory / Metabolism / Governance / Trust core to lab responsibilities.
2. **Ledger Provenance** — Inspect `tessrax/provenance/tracer.py` and reproduce a
   provenance receipt for a lab claim.
3. **Ethical Drift Lab** — Run `sandbox/ethical_drift.py` with deterministic
   seeds to observe drift tolerances.
4. **Human Feedback Loop** — Submit corrections to the `/feedback/` endpoint and
   verify ledger persistence.
5. **Federation Visualization** — Launch `dashboard/federation_map.py` to review
   consensus across lab nodes.
6. **SDK Integration** — Use `sdk/tessrax_client.py` to submit an automated
   claim from lab notebooks.
7. **Contradiction Detection** — Explore `tests/test_provenance_tracer.py` to
   understand deterministic hashing.
8. **Drift Analytics** — Compare successive runs of the ethical drift summary to
   quantify legitimate deviations.
9. **Health Monitoring** — Review `tessrax/diagnostics/health_monitor.py` and the
   GitHub Action badge to understand automated audits.
10. **Reflection & Feedback** — Post learnings via the `/feedback/` API and
    attach sandbox experiment hashes.

## Hands-On Code Labs

- **Lab 1 — Provenance Receipt Builder**: Extend
  `tessrax/provenance/tracer.py` in a notebook to ingest real AI Skills Lab
  outputs, sign them with Ed25519, and store receipts in `out/provenance_receipt.json`.
- **Lab 2 — Ethical Drift Dashboard**: Combine the JSONL receipts from
  `sandbox/ethical_drift.py` with `dashboard/federation_map.py` to produce a
  cross-validated drift vs. integrity chart.

Learners should capture reflections in the ledger by invoking the governed
Human Feedback API once each module concludes.  The sandbox enables safe prompt
experiments while maintaining DLK-compliant provenance trails.
