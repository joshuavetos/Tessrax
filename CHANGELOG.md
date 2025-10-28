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
