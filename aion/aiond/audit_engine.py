"""AION audit engine for Tessrax repositories.

This module provides a deterministic audit routine that produces
receipts compliant with the Tessrax Governance directives. The engine
verifies repository metrics, computes integrity scores, and stores an
attestable receipt file containing runtime verification data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

AUDITOR_IDENTITY = "Tessrax Governance Kernel v16"
GOVERNANCE_CLAUSES = ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"]
DEFAULT_RECEIPT_PATH = Path("out/aion_audit_receipt.json")


def _sha256_hexdigest(payload: Dict[str, Any]) -> str:
    """Return a SHA-256 digest for the provided payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


@dataclass
class AuditMetrics:
    """Holds simple repository metrics used for audit scoring."""

    python_modules: int
    workflows: int
    documentation_files: int

    def integrity_score(self) -> float:
        """Compute a normalized integrity score.

        The function weighs the presence of core repository assets. The
        computation is deterministic and bounded to the [0.0, 1.0]
        interval as mandated by the audit protocol.
        """

        score = 0.0
        score += min(self.python_modules / 25.0, 0.35)
        score += min(self.workflows / 10.0, 0.30)
        score += min(self.documentation_files / 15.0, 0.35)
        return round(min(score, 1.0), 4)


class AuditEngine:
    """Tessrax AION audit executor.

    The engine collects repository metrics, validates them against a
    configurable threshold, and materializes an auditable receipt.
    """

    def __init__(self, repo_root: Path, threshold: float = 0.9) -> None:
        if not repo_root.exists():
            raise FileNotFoundError(f"Repository root {repo_root} does not exist")
        self.repo_root = repo_root.resolve()
        self.threshold = threshold

    def _count_matching(self, *extensions: str, subdir: str | None = None) -> int:
        base = self.repo_root if subdir is None else self.repo_root / subdir
        if not base.exists():
            return 0
        count = 0
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in extensions:
                count += 1
        return count

    def collect_metrics(self) -> AuditMetrics:
        python_modules = self._count_matching(".py")
        workflows = self._count_matching(".yml", ".yaml", subdir=".github/workflows")
        documentation_files = self._count_matching(".md", subdir="docs")
        return AuditMetrics(
            python_modules=python_modules,
            workflows=workflows,
            documentation_files=documentation_files,
        )

    def run(self) -> Dict[str, Any]:
        metrics = self.collect_metrics()
        integrity = metrics.integrity_score()
        status = "pass" if integrity >= self.threshold else "fail"
        receipt: Dict[str, Any] = {
            "timestamp": time.time(),
            "runtime_info": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "repo_root": str(self.repo_root),
            },
            "metrics": asdict(metrics),
            "integrity_score": integrity,
            "status": status,
            "auditor": AUDITOR_IDENTITY,
            "clauses": GOVERNANCE_CLAUSES,
        }
        receipt["signature"] = _sha256_hexdigest(receipt)
        if integrity < self.threshold:
            raise RuntimeError(
                "Integrity threshold not met: "
                f"{integrity} < {self.threshold}. Receipt: {json.dumps(receipt, indent=2)}"
            )
        return receipt


def write_receipt(receipt: Dict[str, Any], path: Path = DEFAULT_RECEIPT_PATH) -> None:
    """Persist the audit receipt to disk in a deterministic manner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> Dict[str, Any]:
    """Entry point used by both CLI and external invocations."""
    parser = argparse.ArgumentParser(description="Execute the Tessrax AION audit engine.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to audit (defaults to current directory).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Integrity threshold for audit pass/fail decisions.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=DEFAULT_RECEIPT_PATH,
        help="Path where the generated receipt will be stored.",
    )
    args = parser.parse_args(argv)

    engine = AuditEngine(repo_root=args.root, threshold=args.threshold)
    receipt = engine.run()
    write_receipt(receipt, path=args.receipt)
    print("DLK-VERIFIED AION audit complete", json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":  # pragma: no cover - CLI behavior
    main()
