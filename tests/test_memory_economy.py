"""Tests for temporal compressor and contradiction economy."""

from __future__ import annotations

import py_compile
import sys
import types
from importlib import util
from pathlib import Path


def _ensure_stub_packages() -> None:
    if "tessrax" not in sys.modules:
        package = types.ModuleType("tessrax")
        package.__path__ = [str(Path("tessrax"))]
        sys.modules["tessrax"] = package
    if "tessrax.core" not in sys.modules:
        core = types.ModuleType("tessrax.core")
        core.__path__ = [str(Path("tessrax/core"))]
        sys.modules["tessrax.core"] = core


def _load_module(name: str, path: Path):
    _ensure_stub_packages()
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


temporal_compressor = _load_module(
    "tessrax.core.memory.temporal_compressor",
    Path("tessrax/core/memory/temporal_compressor.py"),
)
contradiction_economy = _load_module(
    "tessrax.core.metabolism.contradiction_economy",
    Path("tessrax/core/metabolism/contradiction_economy.py"),
)


def test_temporal_compressor_self_test():
    assert temporal_compressor._self_test()


def test_contradiction_economy_self_test():
    assert contradiction_economy._self_test()


def test_py_compile_memory_modules():
    py_compile.compile("tessrax/core/memory/temporal_compressor.py", doraise=True)
    py_compile.compile("tessrax/core/metabolism/contradiction_economy.py", doraise=True)
