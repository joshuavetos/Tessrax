#!/bin/bash
set -e
echo "Committing SAFEPOINT_TRUTH_LOCK_PROTOTYPE_V1..."
git add truth_lock_api.py \
  ledger_persistence.py \
  tests/test_truth_lock_api.py \
  requirements.txt \
  .github/workflows/truth-lock-ci.yml \
  docs/TRUTH_LOCK_README.md \
  scripts/commit_truth_lock_v1.sh
git commit -m "feat(truth-lock): add SAFEPOINT_TRUTH_LOCK_PROTOTYPE_V1"
git tag SAFEPOINT_TRUTH_LOCK_PROTOTYPE_V1
echo "Commit and tag complete."
