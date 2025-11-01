"""Automated Tessrax self-test discovery and execution utilities."""

from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable, List, Sequence

_BASE_PACKAGES: tuple[str, ...] = ("tessrax", "tessrax.core")


@dataclass(frozen=True)
class SelfTestResult:
    """Container capturing execution outcomes for `_self_test` functions."""

    module: str
    succeeded: bool
    message: str


def _resolve_package_paths(package: ModuleType) -> List[Path]:
    if not hasattr(package, "__path__"):
        return []
    return [Path(path) for path in package.__path__]


def _module_name_from_path(base: str, package_path: Path, py_file: Path) -> str:
    relative = py_file.relative_to(package_path)
    parts = relative.with_suffix("").parts
    if not parts:
        return base
    return ".".join((base, *parts))


def _discover_modules(packages: Sequence[str]) -> Iterable[str]:
    seen: set[str] = set()
    for base_name in packages:
        try:
            base_module = importlib.import_module(base_name)
        except Exception as exc:  # pragma: no cover - defensive guard
            yield f"!failed_import:{base_name}:{exc}"
            continue
        module_file = getattr(base_module, "__file__", None)
        if module_file and "_self_test" in Path(module_file).read_text(encoding="utf-8"):
            seen.add(base_name)
            yield base_name
        for package_path in _resolve_package_paths(base_module):
            for py_file in package_path.rglob("*.py"):
                if "__pycache__" in py_file.parts:
                    continue
                try:
                    content = py_file.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                if "_self_test" not in content:
                    continue
                module_name = _module_name_from_path(base_name, package_path, py_file)
                if module_name not in seen:
                    seen.add(module_name)
                    yield module_name


def _execute_self_test(module_name: str) -> SelfTestResult:
    if module_name.startswith("!failed_import:"):
        return SelfTestResult(module=module_name, succeeded=False, message="import failure")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return SelfTestResult(module=module_name, succeeded=False, message=f"import error: {exc}")
    candidate = getattr(module, "_self_test", None)
    if candidate is None:
        return SelfTestResult(module=module_name, succeeded=True, message="no self-test")
    if not callable(candidate):
        return SelfTestResult(module=module_name, succeeded=False, message="_self_test not callable")
    signature = inspect.signature(candidate)
    if signature.parameters:
        return SelfTestResult(module=module_name, succeeded=False, message="_self_test has parameters")
    try:
        outcome = candidate()
    except Exception as exc:  # pragma: no cover - failure path is summarised
        return SelfTestResult(module=module_name, succeeded=False, message=f"raised {exc}")
    if outcome is not True:
        return SelfTestResult(module=module_name, succeeded=False, message=f"returned {outcome!r}")
    return SelfTestResult(module=module_name, succeeded=True, message="passed")


def run_self_tests(packages: Sequence[str] | None = None) -> List[SelfTestResult]:
    """Run `_self_test` hooks for all modules beneath the supplied packages."""

    targets = packages or _BASE_PACKAGES
    results: List[SelfTestResult] = []
    for module_name in _discover_modules(targets):
        results.append(_execute_self_test(module_name))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Execute Tessrax self-tests and emit DLK-verified summary output."""

    packages = tuple(argv[1:]) if argv else _BASE_PACKAGES
    if not packages:
        packages = _BASE_PACKAGES
    results = run_self_tests(packages)
    failures = [result for result in results if not result.succeeded]
    executed = [result for result in results if result.message != "no self-test"]
    print("Tessrax self-test execution summary:")
    for result in executed:
        status = "PASS" if result.succeeded else "FAIL"
        print(f" - {status:<4} {result.module}: {result.message}")
    skipped_count = len(results) - len(executed)
    if skipped_count:
        print(f"Skipped modules without _self_test: {skipped_count}")
    print(
        "Double-lock receipt:",
        {
            "auditor": "Tessrax Governance Kernel v16",
            "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
            "status": "DLK-VERIFIED" if not failures else "DLK-FAILED",
            "integrity_score": 0.99 if not failures else 0.0,
            "runtime_info": {
                "python": sys.version,
                "module_count": len(results),
                "executed": len(executed),
                "failures": len(failures),
            },
            "signature": Path(__file__).resolve().as_posix(),
        },
    )
    return 0 if not failures else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main(sys.argv))
