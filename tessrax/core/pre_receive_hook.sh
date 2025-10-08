#!/bin/bash
# Tessrax Governance Enforcement Hook
# Denies Git commits if Tessrax detects contradictions.

echo "Running Tessrax pre-receive audit..."

# Run the audit script on incoming commit
python3 /data/conflict_graph.py --check .
RESULT=$?

if [ $RESULT -ne 0 ]; then
  echo "❌ Contradiction detected. Commit rejected to preserve governance integrity."
  exit 1
else
  echo "✅ No contradictions found. Commit accepted."
  exit 0
fi