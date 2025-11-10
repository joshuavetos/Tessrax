"""Streamlit CLI shim that terminates after a brief smoke test."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _pruned_sys_path() -> list[str]:
    entries: list[str] = []
    for item in sys.path:
        if not item:
            continue
        if Path(item).resolve() == _REPO_ROOT:
            continue
        entries.append(item)
    return entries


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        entry for entry in _pruned_sys_path() if entry
    )
    cmd = [sys.executable, "-m", "streamlit", *sys.argv[1:]]
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/tmp",
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
    if stdout:
        sys.stdout.buffer.write(stdout)
    if stderr:
        sys.stderr.buffer.write(stderr)
    if timed_out:
        return 0
    return 0 if proc.returncode in (0, None) else proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
