# Changelog

## Runtime Stabilization v1.1
- Locked the supported runtime to Python 3.10.x across CI and package metadata.
- Established `requirements.txt`/`requirements-lock.txt` workflow with automated CI verification.
- Normalised imports to absolute `tessrax.*` paths and added lint rules against regressions.
- Added placeholder ledger and CorporateFrienthropy datasets for safe local execution.
- Documented environment setup, verification steps, and dependency policy updates.

## Repository Audit Cleanup v1.2
- Removed the legacy Truth-Lock prototype service, dedicated CI workflow, and associated tests in favour of the Tessrax Truth API.
- Deleted archived redundancy audit artefacts and helper scripts that were no longer part of the active release process.

## Tessrax v17.5 — Governed Ethical Drift Engine + Pedagogical Layer
- Added DLK-verified ethical drift engine with receipt generation and entropy safeguards.
- Authored governed AI literacy curriculum v1 linking sandbox demos for Modules 5 and 6.
- Introduced reusable skills lab sandboxes aligned with governed audit outputs.

## Tessrax v17.4 — Governed Provenance + Feedback Suite
- Added DLK-verified provenance tracing with Ed25519 signing in `tessrax/provenance/tracer.py` and regression tests.
- Introduced the ethical drift simulator sandbox with reproducible JSON receipts and summaries.
- Exposed a governed Human Feedback FastAPI router with rate limiting and ledger receipts, including documentation.
- Delivered a Streamlit + D3 federation visualiser with accessible colour palette and PNG snapshot export.
- Published a safe SDK client library with retry logic, typed exceptions, and documentation for integrators.
- Authored a pedagogy curriculum aligning AI Skills Lab modules with Tessrax governed workflows.
- Implemented the auto-governance health monitor, tests, and GitHub Action with dashboard badge output.
