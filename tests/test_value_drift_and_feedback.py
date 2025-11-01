"""Tests for value drift monitor and human feedback API."""
from __future__ import annotations

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


value_drift_monitor = _load_module(
    "tessrax.core.ethics.value_drift_monitor", Path("tessrax/core/ethics/value_drift_monitor.py")
)
human_feedback = _load_module(
    "tessrax.api.human_feedback", Path("tessrax/api/human_feedback.py")
)


def test_value_drift_self_test():
    assert value_drift_monitor._self_test()


def test_human_feedback_roundtrip(tmp_path):
    # Backup existing file
    data_path = Path(human_feedback.DATA_PATH)
    original = data_path.read_text(encoding="utf-8") if data_path.exists() else ""
    try:
        assert human_feedback._self_test()
        history = human_feedback.get_history()["history"]
        assert history, "History should contain entries"
        last_entry = history[-1]
        assert last_entry["verdict"], "Verdict should be recorded"
    finally:
        data_path.write_text(original, encoding="utf-8")
