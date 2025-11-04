"""Module 6 sandbox focusing on ethical drift dynamics and entropy monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from tessrax.ethics.ethical_drift_engine import run_with_audit


def run_module6_demo(output_directory: Path | str = Path("out/module6")) -> Tuple[Path, Path]:
    """Execute a high-resolution ethical drift simulation for Module 6 learners."""

    return run_with_audit(runs=75, output_directory=output_directory)
