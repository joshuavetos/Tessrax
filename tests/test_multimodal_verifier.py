"""Tests for multimodal verifier module."""
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


verifier = _load_module("tessrax.core.multimodal.verifier", Path("tessrax/core/multimodal/verifier.py"))


def test_multimodal_self_test():
    assert verifier._self_test()


def test_alignment_deterministic():
    claim = "Solar array produces 5kW under clear skies"
    caption = "Image shows solar array producing 5kW"
    transcript = "Audio confirms solar array produces five kilowatts"
    metrics_first = verifier.verify_alignment(claim, caption, transcript)
    metrics_second = verifier.verify_alignment(claim, caption, transcript)
    assert metrics_first == metrics_second
    assert 0.0 <= metrics_first["consistency_score"] <= 1.0


def test_py_compile_verifier():
    py_compile.compile("tessrax/core/multimodal/verifier.py", doraise=True)
