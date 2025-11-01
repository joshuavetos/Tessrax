"""Tests for TEX protocol and policy compiler modules."""

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


policy_compiler = _load_module(
    "tessrax.core.governance.policy_compiler",
    Path("tessrax/core/governance/policy_compiler.py"),
)
tex_protocol = _load_module(
    "tessrax.core.protocols.tex_protocol",
    Path("tessrax/core/protocols/tex_protocol.py"),
)
audit_integration = _load_module(
    "tessrax.core.ledger.audit_integration",
    Path("tessrax/core/ledger/audit_integration.py"),
)


def test_tex_protocol_self_test():
    assert tex_protocol._self_test()


def test_policy_compiler_self_test():
    assert policy_compiler._self_test()


def test_policy_compiler_evaluation_consistency():
    laws = policy_compiler.compile_iron_laws(["Iron Law: solar -> verified @0.9"])
    claim = {"text": "Solar panels verified", "confidence": 0.82}
    evaluations = policy_compiler.evaluate_claim(claim, laws)
    assert evaluations[0]["truth_status"] == "verified"
    assert evaluations[0]["confidence"] == 0.82


def test_py_compile_protocol_and_policy():
    py_compile.compile("tessrax/core/protocols/tex_protocol.py", doraise=True)
    py_compile.compile("tessrax/core/governance/policy_compiler.py", doraise=True)


def test_audit_integration_self_test(tmp_path):
    ledger_path = Path(audit_integration.LEDGER_PATH)
    original = ledger_path.read_text(encoding="utf-8") if ledger_path.exists() else ""
    try:
        assert audit_integration._self_test()
        verification = audit_integration.verify_audit_receipt(
            audit_integration.append_audit_receipt({"title": "T", "auditor": "A"})[
                "receipt_id"
            ]
        )
        assert verification["legitimacy"] >= 0.8
    finally:
        ledger_path.write_text(original, encoding="utf-8")
