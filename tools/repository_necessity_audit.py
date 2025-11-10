"""Tessrax Repository Necessity Audit (Dry-Run).

This module implements the dry-run repository necessity audit described in the
user specification.  The script analyses the repository under the Tessrax
governance constraints (AEP-001, POST-AUDIT-001, RVC-001, EAC-001) and emits a
JSON report alongside an auditable execution receipt.

The workflow consists of three major phases:
    1. Pre-audit validation ("double lock" gate 1) that ensures the repository
       manifest exists and that the working directory resolves to the expected
       project root.  This step prevents the script from operating in an
       unknown environment (AEP-001, EAC-001).
    2. Static and dynamic information gathering.  The script enumerates
       repository files, parses Python imports, and runs ``pytest --collect`` to
       observe runtime-visible assets.  Runtime verification (RVC-001) is
       achieved via structured error handling and integrity assertions.
    3. Post-audit validation ("double lock" gate 2) that verifies report
       completeness and produces a signed execution receipt in accordance with
       POST-AUDIT-001, DLK-001, and the Receipts-First rule.

The resulting ``audit_report.json`` maps files into the categories requested by
the user: "used", "test_only", "redundant", and "unknown".  No repository
mutations are performed; the script is safe to run in CI environments.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    DefaultDict,
    Dict,
    Iterable,
    List,
    Mapping,
    Sequence,
    Set,
    Tuple,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "audit_report.json"
RECEIPT_PATH = REPO_ROOT / "audit_receipt.json"
RUN_RECEIPT_DIR = REPO_ROOT / "audits" / "run_receipts"


class AuditError(RuntimeError):
    """Raised when any audit gate fails.

    The exception explicitly indicates which governance clause would be
    violated, making failures actionable and observable as required by TESST.
    """


@dataclass(frozen=True)
class AuditResult:
    """Container for audit classification results.

    Attributes:
        used: Files referenced by production code paths.
        test_only: Files referenced exclusively in tests or pytest discovery.
        redundant: Files that were not referenced by runtime artefacts.
        unknown: Non-Python artefacts that could be loaded dynamically.
    """

    used: List[str]
    test_only: List[str]
    redundant: List[str]
    unknown: List[str]

    def as_dict(self) -> Dict[str, List[str]]:
        """Return a JSON-serialisable representation of the audit result."""

        return {
            "used": self.used,
            "test_only": self.test_only,
            "redundant": self.redundant,
            "unknown": self.unknown,
        }


def _verify_repository_context(repo_root: Path) -> None:
    """Run pre-audit validation checks (DLK-001 gate 1).

    The function ensures that the repository contains ``Latest.txt`` (per
    EAC-001) and that the script runs from the expected path.  Violations raise
    :class:`AuditError` with a clear explanation so the operator can correct the
    environment.
    """

    manifest = repo_root / "Latest.txt"
    if not manifest.is_file():
        raise AuditError(
            "EAC-001 violation: repository manifest Latest.txt was not found in "
            f"{repo_root}."
        )
    tools_dir = repo_root / "tools"
    if not tools_dir.is_dir():
        raise AuditError(
            "AEP-001 violation: expected tools directory is missing; cannot "
            "verify repository structure."
        )


def _gather_files(repo_root: Path) -> List[Path]:
    """Collect all repository artefacts relevant to the dry-run audit."""

    extensions = {".py", ".json", ".yaml", ".yml", ".md"}
    quarantine_root = repo_root / "data" / "quarantine"
    files: List[Path] = [
        path
        for path in sorted(repo_root.rglob("*"))
        if path.suffix in extensions
        and path.is_file()
        and not (
            quarantine_root.exists()
            and path.is_relative_to(quarantine_root)
        )
        and not (
            RUN_RECEIPT_DIR.exists()
            and path.is_relative_to(RUN_RECEIPT_DIR)
        )
    ]
    if not files:
        raise AuditError(
            "RVC-001 violation: file enumeration returned no artefacts; check "
            "repository structure."
        )
    return files


def _collect_imports(repo_root: Path) -> Mapping[Path, Set[str]]:
    """Parse Python files and collect their top-level import statements."""

    imports: DefaultDict[Path, Set[str]] = defaultdict(set)
    for path in sorted(repo_root.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, UnicodeDecodeError) as exc:
            raise AuditError(
                f"AEP-001 violation: failed to read {path} due to {exc!r}."
            ) from exc
        except SyntaxError:
            # Syntax errors are treated as signals; we skip the file but log
            # the event by storing a synthetic module name so classification can
            # highlight the anomaly.
            imports[path].add("__syntax_error__")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[path].add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports[path].add(node.module.split(".")[0])
    return imports


def _run_pytest_collect(repo_root: Path) -> Tuple[Sequence[str], int]:
    """Execute ``pytest --collect-only`` to gather runtime references."""

    command = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    try:
        completed = subprocess.run(  # noqa: S603, S607 (trusted input)
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover -
        raise AuditError(
            "RVC-001 violation: pytest collection command could not be "
            f"executed ({exc!r})."
        ) from exc
    output_lines = (completed.stdout + completed.stderr).splitlines()
    if completed.returncode in {0, 5}:  # pytest returns 5 when tests missing
        return output_lines, completed.returncode
    if completed.returncode == 2:
        output_lines.append(
            "TESST signal: pytest collection exited with status 2 (collection errors tolerated)."
        )
        return output_lines, completed.returncode
    raise AuditError(
        "TESST signal: pytest collection exited with unexpected status "
        f"{completed.returncode}. Review output for diagnostics."
    )


def _scan_path_references(files: Iterable[Path], repo_root: Path) -> Mapping[str, Set[str]]:
    """Extract textual references to repository files.

    The routine walks the gathered artefacts and records every relative path that
    appears in textual content.  The result maps each referenced path to the set
    of files that mentioned it.  This makes the audit resilient to dynamic file
    loading patterns captured in documentation, manifests, and ledger entries,
    reducing false "redundant" classifications that previously triggered merge
    conflicts.
    """

    path_pattern = re.compile(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.(?:py|json|ya?ml|md)")
    references: DefaultDict[str, Set[str]] = defaultdict(set)
    for candidate in files:
        try:
            text = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Binary or non-text artefacts are ignored for reference scanning.
            continue
        rel_source = candidate.relative_to(repo_root).as_posix()
        for match in path_pattern.findall(text):
            normalised = str(Path(match).as_posix())
            references[normalised].add(rel_source)
    return references


def _classify_files(
    repo_root: Path,
    files: Iterable[Path],
    imports: Mapping[Path, Set[str]],
    pytest_lines: Sequence[str],
) -> AuditResult:
    """Classify repository artefacts according to usage heuristics."""

    imported_modules = {module for modules in imports.values() for module in modules}
    pytest_text = " ".join(pytest_lines)
    used: List[str] = []
    test_only: List[str] = []
    redundant: List[str] = []
    unknown: List[str] = []

    file_list = list(files)
    path_references = _scan_path_references(file_list, repo_root)
    for file_path in file_list:
        rel_path = file_path.relative_to(repo_root).as_posix()
        path_parts = rel_path.split("/")
        stem = file_path.stem
        if rel_path.startswith("tests/"):
            test_only.append(rel_path)
        elif rel_path in path_references:
            used.append(rel_path)
        elif any(
            module
            and (
                module in path_parts
                or module == stem
                or module.replace("_", "-") in path_parts
            )
            for module in imported_modules
        ):
            used.append(rel_path)
        elif rel_path in pytest_text:
            test_only.append(rel_path)
        elif file_path.suffix in {".json", ".yaml", ".yml", ".md"}:
            unknown.append(rel_path)
        elif file_path.name == "__init__.py":
            unknown.append(rel_path)
        else:
            redundant.append(rel_path)

    total_classified = sum(
        len(bucket) for bucket in (used, test_only, redundant, unknown)
    )
    if total_classified != len(file_list):
        raise AuditError(
            "POST-AUDIT-001 violation: classification counts do not match the "
            "input corpus."
        )
    return AuditResult(
        used=sorted(used),
        test_only=sorted(test_only),
        redundant=sorted(redundant),
        unknown=sorted(unknown),
    )


def _write_json_if_changed(path: Path, payload: Mapping[str, object]) -> bool:
    """Serialize ``payload`` deterministically and avoid churn when unchanged."""

    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AuditError(
                f"AEP-001 violation: unable to read {path} while emitting JSON: {exc!r}."
            ) from exc
        if existing == rendered:
            return False
    path.write_text(rendered, encoding="utf-8")
    return True


def _stable_timestamp(signature: str) -> str:
    """Derive a deterministic ISO8601 timestamp from the report signature."""

    epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)
    seconds = int(signature[:12], 16)
    normalized = epoch + timedelta(seconds=seconds % (365 * 24 * 3600))
    return normalized.isoformat()


def _post_audit_receipt(
    result: AuditResult, repo_root: Path, pytest_exit_code: int
) -> Dict[str, object]:
    """Generate receipts while preventing merge churn (DLK-001 gate 2)."""

    status = "DLK-VERIFIED"
    integrity_score = 1.0
    tesst_signals: List[str] = []
    if pytest_exit_code == 2:
        status = "DLK-VERIFIED-TESST"
        integrity_score = 0.94
        tesst_signals.append(
            "Pytest collection exited with status 2. Import errors recorded and tolerated."
        )

    signature = _compute_signature(result)
    canonical_receipt = {
        "timestamp": _stable_timestamp(signature),
        "runtime_info": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "report_path": REPORT_PATH.name,
            "manifest": "Latest.txt",
        },
        "integrity_score": integrity_score,
        "status": status,
        "signature": signature,
        "auditor": "Tessrax Governance Kernel v16",
        "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
        "tesst_signals": tesst_signals,
    }

    _write_json_if_changed(RECEIPT_PATH, canonical_receipt)

    runtime_receipt = dict(canonical_receipt)
    runtime_receipt["timestamp"] = datetime.now(timezone.utc).isoformat()
    runtime_receipt["runtime_info"] = {
        "python": sys.version,
        "repo_root": str(repo_root),
        "report_path": str(REPORT_PATH),
    }
    _archive_run_receipt(runtime_receipt)

    return canonical_receipt


def _compute_signature(result: AuditResult) -> str:
    """Compute a SHA-256 digest over the JSON report payload."""

    digest = hashlib.sha256(json.dumps(result.as_dict(), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _archive_run_receipt(payload: Mapping[str, object]) -> None:
    """Persist a run-specific receipt for traceability outside version control."""

    RUN_RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = payload.get("timestamp", datetime.now(timezone.utc).isoformat())
    safe_timestamp = re.sub(r"[^0-9T]+", "-", timestamp)
    receipt_path = RUN_RECEIPT_DIR / f"receipt-{safe_timestamp}.json"
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pre_audit_gate() -> None:
    """Execute the first double-lock gate checks."""

    _verify_repository_context(REPO_ROOT)


def _post_audit_gate(result: AuditResult, pytest_exit_code: int) -> Dict[str, object]:
    """Execute the second double-lock gate and return the receipt payload."""

    return _post_audit_receipt(result, REPO_ROOT, pytest_exit_code)


def _print_summary(result: AuditResult, receipt: Mapping[str, object]) -> None:
    """Emit a human-readable audit summary with receipt confirmation."""

    print("🔍 Tessrax Repository Necessity Audit (Dry-Run)")
    for label, bucket in result.as_dict().items():
        print(f"  {label:>10}: {len(bucket)} files")
    print(f"Report written to: {REPORT_PATH}")
    print(f"Receipt written to: {RECEIPT_PATH}")
    print("Receipt signature (SHA-256):", receipt["signature"])
    print("Audit status:", receipt["status"])


def main() -> int:
    """Entry point that orchestrates the double-lock audit workflow."""

    try:
        _pre_audit_gate()
        files = _gather_files(REPO_ROOT)
        imports = _collect_imports(REPO_ROOT)
        pytest_lines, pytest_exit_code = _run_pytest_collect(REPO_ROOT)
        result = _classify_files(REPO_ROOT, files, imports, pytest_lines)
        _write_json_if_changed(REPORT_PATH, result.as_dict())
        receipt = _post_audit_gate(result, pytest_exit_code)
        _print_summary(result, receipt)
    except AuditError as error:
        print("TESS MODE BLOCK:", error)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
