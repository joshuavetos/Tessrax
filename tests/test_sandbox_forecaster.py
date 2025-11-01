"""Tests for adversarial sandbox and causal forecaster modules."""

from __future__ import annotations

import py_compile
import sys
import types
from importlib import util
from pathlib import Path
from time import perf_counter


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


adversarial_sim = _load_module(
    "tessrax.core.sandbox.adversarial_sim",
    Path("tessrax/core/sandbox/adversarial_sim.py"),
)
causal_forecaster = _load_module(
    "tessrax.core.predictive.causal_forecaster",
    Path("tessrax/core/predictive/causal_forecaster.py"),
)


def test_adversarial_self_test_passes():
    assert adversarial_sim._self_test()
    metrics = adversarial_sim.evaluate_recovery()
    assert metrics["accuracy"] >= 0.9


def test_causal_forecaster_distribution_and_runtime():
    sample = [
        {"id": "alpha", "novelty": 0.2, "support": 0.9},
        {"id": "beta", "novelty": 0.6, "support": 0.6},
        {"id": "gamma", "novelty": 0.8, "support": 0.4, "penalty": 0.05},
    ]
    start = perf_counter()
    distribution = causal_forecaster.forecast_contradictions(sample)
    runtime = perf_counter() - start
    assert abs(sum(distribution.values()) - 1.0) < 1e-6
    assert runtime < 5.0


def test_py_compile_integrity():
    for module in [
        Path("tessrax/core/sandbox/adversarial_sim.py"),
        Path("tessrax/core/predictive/causal_forecaster.py"),
    ]:
        py_compile.compile(str(module), doraise=True)
