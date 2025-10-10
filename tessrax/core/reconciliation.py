"""
Reconciliation Module
Combines diff_tool.py and fork_reconciliation_engine.py.
Provides utilities for detecting and resolving content drift.
"""

import difflib
from pathlib import Path


def diff_files(file_a: str, file_b: str) -> str:
    """Return a unified diff string between two files."""
    with open(file_a) as fa, open(file_b) as fb:
        diff = difflib.unified_diff(
            fa.readlines(), fb.readlines(),
            fromfile=file_a, tofile=file_b, lineterm=""
        )
    return "\n".join(diff)


def reconcile(source_path: str, target_path: str, output_path: str):
    """Simple line-based reconciliation: favor source on conflict."""
    with open(source_path) as src, open(target_path) as tgt:
        merged = []
        for a, b in zip(src, tgt):
            merged.append(a if a != b else a)
    Path(output_path).write_text("".join(merged))


def find_conflicts(base: str, compare: str):
    """Identify lines that differ."""
    with open(base) as f1, open(compare) as f2:
        lines1, lines2 = f1.readlines(), f2.readlines()
    return [i for i, (a, b) in enumerate(zip(lines1, lines2), 1) if a != b]
