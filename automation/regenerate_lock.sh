#!/usr/bin/env bash
set -euo pipefail

python automation/meta_launcher/hooks/check_python_sync.py || {
  echo "Python interpreter version mismatch detected — aborting lock-file regeneration."
  exit 1
}

pip install -r requirements.txt
pip freeze > requirements-lock.txt
