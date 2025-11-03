"""End-to-end smoketest validating Tessrax core runtime surface."""

from __future__ import annotations

import importlib

from tessrax.selftest import run_self_tests


def test_selftests_execute_without_failures() -> None:
    """Run Tessrax self-tests and assert the DLK gate stays green."""

    results = run_self_tests()
    failures = [result for result in results if not result.succeeded]
    assert not failures, f"Self-test failures detected: {failures!r}"


def test_ed25519_backward_compatibility_alias() -> None:
    """Ensure legacy `ed25519` imports remain operational via PyNaCl alias."""

    module = importlib.import_module("ed25519")
    signing_key = module.SigningKey.generate()
    message = b"tessrax-governance"
    signature = signing_key.sign(message)
    verify_key = module.VerifyKey(signing_key.verify_key.encode())
    verify_key.verify(message, signature.signature)
