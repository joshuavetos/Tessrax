"""
Python Sync Guard — prevents dependency lock corruption.
Ensures that the active Python interpreter version locally matches the declared
version in the CI pipeline (.github/workflows/tests.yml) before allowing
regeneration of requirements-lock.txt.

Usage:
Run this script as a pre-check in automation workflows or locally before
dependency freezing commands.

Authorizing Architect: Josh Scott Vetos
Patch ID: SAFEPOINT_V15_1_PYTHON_SYNC_GUARD
"""

import datetime
import json
import pathlib
import re
import sys

import yaml

WORKFLOW_FILE = pathlib.Path(".github/workflows/tests.yml")
LEDGER_FILE = pathlib.Path("automation/meta_launcher/logs/ledger.log")


def read_declared_python_version():
    if not WORKFLOW_FILE.exists():
        print("[PYTHON SYNC GUARD] ⚠️ Workflow file not found at", WORKFLOW_FILE)
        sys.exit("Critical: Workflow file missing, aborting to prevent desync.")
    try:
        data = yaml.safe_load(WORKFLOW_FILE.read_text())
        jobs = data.get("jobs", {}) or {}
        # Prefer 'build' but fall back to the first job defined.
        job_config = jobs.get("build")
        if job_config is None and jobs:
            job_config = next(iter(jobs.values()))
        if not isinstance(job_config, dict):
            sys.exit("Critical: No usable job definition found in workflow.")
        matrix = job_config.get("strategy", {}).get("matrix", {})
        versions = matrix.get("python-version", [])
        if not versions:
            steps = job_config.get("steps", []) or []
            for step in steps:
                if not isinstance(step, dict):
                    continue
                if "actions/setup-python" in str(step.get("uses", "")):
                    versions = step.get("with", {}).get("python-version")
                    break
        if not versions:
            sys.exit("Critical: No python-version in CI workflow, aborting.")
        version_candidate = versions if isinstance(versions, str) else versions[0]
        match = re.match(r"(\d+\.\d+)", str(version_candidate))
        if not match:
            sys.exit(
                f"Critical: Unable to parse Python version from '{version_candidate}'."
            )
        return match.group(1)
    except Exception as e:
        sys.exit(f"[PYTHON SYNC GUARD] Error reading workflow file: {e}")


def get_local_python_version():
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def log_guard_result(status: str, declared: str, local: str) -> None:
    try:
        LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry = {
            "timestamp": timestamp,
            "directive": "SAFEPOINT_V15_1_PYTHON_SYNC_GUARD",
            "summary": (
                "Python Sync Guard check executed. CI declared version="
                f"{declared}, local interpreter={local}."
            ),
            "status": status,
        }
        with LEDGER_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except Exception as exc:  # pragma: no cover - best-effort logging
        print(f"[PYTHON SYNC GUARD] ⚠️ Failed to append ledger entry: {exc}")


def main():
    declared = read_declared_python_version()
    local = get_local_python_version()

    if local != declared:
        print("[PYTHON SYNC GUARD] ❌ Python version mismatch detected!")
        print(f"  - CI matrix declared version: Python {declared}")
        print(f"  - Local environment version:  Python {local}")
        print(
            "  Please activate a Python interpreter matching the CI version before proceeding."
        )
        log_guard_result("failure", declared, local)
        sys.exit(1)

    print(
        f"[PYTHON SYNC GUARD] ✅ Local Python {local} matches CI matrix. Safe to proceed."
    )
    log_guard_result("success", declared, local)


if __name__ == "__main__":
    main()
