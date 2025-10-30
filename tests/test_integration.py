"""Full end-to-end validation of the CLI workflow."""

from __future__ import annotations

import json
import subprocess
import sys


def test_cli_render_command(tmp_path) -> None:
    """Running the render command should return the expected template."""

    command = [
        sys.executable,
        "-m",
        "ai_skills.prompting.cli",
        "render",
        "--template",
        "socratic_debugger",
        "--task",
        "Add numbers",
        "--context",
        "3 and 5",
    ]
    output = subprocess.check_output(command, text=True)
    assert "TASK: Add numbers" in output
    assert "CONTEXT: 3 and 5" in output


def test_cli_score_command(tmp_path) -> None:
    """Running the score command should produce JSON with match information."""

    command = [
        sys.executable,
        "-m",
        "ai_skills.prompting.cli",
        "score",
        "--guess",
        "test",
        "--truth",
        "test",
    ]
    output = subprocess.check_output(command, text=True)
    payload = json.loads(output)
    assert payload["match"] is True
    assert payload["similarity"] == 1.0
