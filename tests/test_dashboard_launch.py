"""Runtime verification for the Tessrax governance dashboard."""
from __future__ import annotations

import shutil
import subprocess
import time

import pytest


def test_dashboard_runs() -> None:
    """Ensure the Streamlit dashboard starts successfully on the configured port."""
    if shutil.which("streamlit") is None:
        pytest.skip("streamlit CLI not available in test environment")

    command = [
        "streamlit",
        "run",
        "dashboard/ledger_viewer.py",
        "--server.port=8090",
        "--server.headless=true",
    ]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        time.sleep(3)
        assert proc.poll() is None, "Dashboard terminated unexpectedly during startup"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    assert proc.returncode is not None
