from __future__ import annotations

from pathlib import Path

import pytest

from tessrax.plugins.sandbox import CPU_LIMIT_SECONDS, MEMORY_LIMIT_BYTES, execute_plugin


def test_plugin_executes_with_math_payload(tmp_path) -> None:
    code = """
result = payload["value"] + 2
"""
    output = execute_plugin(code, payload={"value": 40})
    assert output == 42


def test_plugin_disallows_imports() -> None:
    code = """
import os
result = 1
"""
    with pytest.raises(ImportError):
        execute_plugin(code)


def test_plugin_restricts_filesystem(tmp_path) -> None:
    code = """
with open('/etc/passwd') as handle:
    handle.read()
result = True
"""
    with pytest.raises(PermissionError):
        execute_plugin(code)


def test_plugin_requires_result_variable() -> None:
    code = """
x = 1 + 1
"""
    with pytest.raises(RuntimeError):
        execute_plugin(code)


def test_v15_plugin_compatibility() -> None:
    plugin_code = Path("tests/data/plugins/v15_example.py").read_text(encoding="utf-8")
    output = execute_plugin(plugin_code, payload={"claims": [1, 2, 3]})
    assert output == {"total": 6, "count": 3}


def test_resource_limits_exposed() -> None:
    assert MEMORY_LIMIT_BYTES == 100 * 1024 * 1024
    assert CPU_LIMIT_SECONDS == 30
