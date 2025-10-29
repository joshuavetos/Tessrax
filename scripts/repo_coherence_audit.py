"""Repository coherence audit utility.

This script indexes Python modules in the Tessrax repository, validates
that they can be imported, and generates dependency metadata to help
identify orphaned or problematic modules. Outputs are written to the
``out`` directory relative to the repository root.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
from typing import List, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "out"


def iter_python_modules(root: Path) -> List[Path]:
    """Return all Python module paths excluding test modules."""
    modules: List[Path] = []
    for path in root.rglob("*.py"):
        if path.name.startswith("test_"):
            continue
        modules.append(path)
    return sorted(modules)


def module_name_for_path(path: Path, root: Path) -> str:
    """Convert a module path to dotted module notation."""
    rel_path = path.relative_to(root).with_suffix("")
    return ".".join(rel_path.parts)


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(exist_ok=True)


def write_module_index(modules: Sequence[Path]) -> None:
    target = OUT_DIR / "module_index.txt"
    lines = [str(path.relative_to(REPO_ROOT)) for path in modules]
    target.write_text("\n".join(lines), encoding="utf-8")


def attempt_import(module_path: Path, module_name: str) -> Tuple[bool, str | None]:
    """Attempt to import the module at ``module_path`` under ``module_name``."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return False, "Unable to load module spec"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - failure path logging only
        return False, str(exc)
    return True, None


def validate_imports(modules: Sequence[Path]) -> List[Tuple[str, str]]:
    failures: List[Tuple[str, str]] = []
    for module_path in modules:
        module_name = module_name_for_path(module_path, REPO_ROOT)
        ok, message = attempt_import(module_path, module_name)
        if not ok and message is not None:
            failures.append((module_name, message))
    return failures


def write_import_failures(failures: Sequence[Tuple[str, str]]) -> None:
    target = OUT_DIR / "import_failures.txt"
    if not failures:
        if target.exists():
            target.unlink()
        return
    formatted = [f"{name}: {error}" for name, error in failures]
    target.write_text("\n".join(formatted), encoding="utf-8")


def build_dependency_edges(modules: Sequence[Path]) -> List[Tuple[str, str]]:
    edges: Set[Tuple[str, str]] = set()
    for module_path in modules:
        source = str(module_path.relative_to(REPO_ROOT))
        try:
            tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=source)
        except (SyntaxError, UnicodeDecodeError):
            # Modules that fail to parse will already appear in the import
            # failure log; skip them when building dependency edges so the
            # audit can continue for the remaining files.
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    edges.add((source, alias.name))
            elif isinstance(node, ast.ImportFrom) and node.module:
                edges.add((source, node.module))
    return sorted(edges)


def write_dependency_edges(edges: Sequence[Tuple[str, str]]) -> None:
    target = OUT_DIR / "dependency_edges.json"
    json_edges = [[src, dst] for src, dst in edges]
    target.write_text(json.dumps(json_edges, indent=2), encoding="utf-8")


def detect_orphans(modules: Sequence[Path], edges: Sequence[Tuple[str, str]]) -> List[Path]:
    imported_modules: Set[str] = {dst for _, dst in edges}
    orphans: List[Path] = []
    for module_path in modules:
        if module_path.name == "__init__.py":
            continue
        mod_name = module_name_for_path(module_path, REPO_ROOT)
        if mod_name not in imported_modules:
            orphans.append(module_path)
    return sorted(orphans)


def write_orphans(orphans: Sequence[Path]) -> None:
    target = OUT_DIR / "orphan_modules.txt"
    lines = [str(path.relative_to(REPO_ROOT)) for path in orphans]
    target.write_text("\n".join(lines), encoding="utf-8")


def write_summary(total_modules: int, failures: Sequence[Tuple[str, str]], orphans: Sequence[Path]) -> None:
    target = OUT_DIR / "repo_coherence_report.json"
    report = {
        "total_modules": total_modules,
        "import_failures": len(failures),
        "orphans_detected": len(orphans),
        "orphan_list": [str(path.relative_to(REPO_ROOT)) for path in orphans],
    }
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")


def perform_audit(root: Path) -> None:
    ensure_out_dir()
    modules = iter_python_modules(root)
    print(f"✅ Indexed {len(modules)} Python modules.")
    write_module_index(modules)

    failures = validate_imports(modules)
    if failures:
        print(f"❌ {len(failures)} import errors logged.")
    else:
        print("✅ All modules import cleanly.")
    write_import_failures(failures)

    edges = build_dependency_edges(modules)
    write_dependency_edges(edges)

    orphans = detect_orphans(modules, edges)
    print(f"⚙️  Found {len(orphans)} potentially unused modules.")
    write_orphans(orphans)

    write_summary(len(modules), failures, orphans)
    print("📄  Repo coherence report generated: out/repo_coherence_report.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Python modules for coherence and connectivity.")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to scan (defaults to repository root deduced from script location).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.root.resolve()
    perform_audit(root)


if __name__ == "__main__":
    main()
