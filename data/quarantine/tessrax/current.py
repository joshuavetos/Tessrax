"""Tessrax runtime entry point invoking the Cold Agent reference run."""

from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Prevent name collisions with the stdlib ``types`` module when importing json.
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

from tessrax.cold_agent.bench import main as run_cold_agent


def main() -> None:
    """Execute the Cold Agent runtime and print the resulting receipt."""

    receipt = run_cold_agent()
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - script execution
    main()
