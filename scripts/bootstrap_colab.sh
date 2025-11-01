#!/usr/bin/env bash
# Tessrax Colab bootstrap script enforcing governed installation steps.
set -euo pipefail

COLAB_ROOT="${COLAB_TESSRAX_DIR:-/content/Tessrax-main}"
if [ ! -d "${COLAB_ROOT}" ]; then
  echo "[Tessrax] Expected repository at ${COLAB_ROOT} but directory is missing." >&2
  exit 1
fi

cd "${COLAB_ROOT}"

pip install -e .
python scripts/run_selftests.py
