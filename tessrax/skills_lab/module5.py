"""Module 5 sandbox bridging human feedback governance with drift auditing."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from tessrax.ethics.ethical_drift_engine import run_with_audit


def run_module5_demo(output_directory: Path | str = Path("out/module5")) -> Tuple[Path, Path]:
    """Execute a reduced-run ethical drift simulation for Module 5 learners."""

    return run_with_audit(runs=25, output_directory=output_directory)
