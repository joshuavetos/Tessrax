# Tessrax v15 → v16 Enterprise Migration Guide

This guide documents the supported path for enterprise operators during the v16
hardening cycle while 3.10 compatibility remains available in the `v15-stable`
tag.

## Runtime Compatibility

- v15 artifacts continue to support Python 3.10 for one release cycle.
- v16 core runtime targets Python 3.11.  Attempting to install on Python 3.10
  emits a deprecation warning advising the use of the `v15-stable` tag.

## Installation Matrix

| Deployment | Command |
|------------|---------|
| Core only  | `pip install tessrax` |
| Enterprise | `pip install tessrax[enterprise]` |
| Legacy     | `pip install tessrax==0.15.*` |

## Plugin Compatibility

Existing v15 plugins remain compatible under the v16 sandbox provided they avoid
prohibited operations (network, filesystem outside `/tmp/plugin_*`).  The test
suite includes a compatibility fixture (`tests/test_plugin_sandbox.py`)
that executes a v15-style plugin module to confirm behaviour.

## Rollback Procedure

1. Checkout the `v15-stable` tag.
2. Reinstall dependencies using Python 3.10.
3. Restore the `out/env_snapshot.json` baseline from your previous audit.
4. Re-run the ledger contract tests to confirm canonical receipts.

Document any rollback decision in the governance ledger with reason codes and
signatures, referencing `HARDENING_V16_COMPLETE` once the forward upgrade is
restored.
