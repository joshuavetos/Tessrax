"""Validate tessrax.selftest module name resolution per governance clauses."""

from __future__ import annotations

from pathlib import Path

from tessrax import selftest


def test_module_name_from_path_root_file(tmp_path: Path) -> None:
    """Ensure root-level modules resolve to their base package name under AEP-001."""

    package_path = tmp_path / "tessrax"
    package_path.mkdir()
    py_file = package_path / "__init__.py"
    py_file.write_text("_self_test = True\n", encoding="utf-8")

    resolved = selftest._module_name_from_path("tessrax", package_path, py_file)

    assert resolved == "tessrax"


def test_module_name_from_path_nested_module(tmp_path: Path) -> None:
    """Ensure nested modules resolve with dotted path while satisfying RVC-001."""

    package_path = tmp_path / "tessrax"
    nested_pkg = package_path / "core"
    nested_pkg.mkdir(parents=True)
    py_file = nested_pkg / "metabolism.py"
    py_file.write_text("_self_test = True\n", encoding="utf-8")

    resolved = selftest._module_name_from_path("tessrax", package_path, py_file)

    assert resolved == "tessrax.core.metabolism"
