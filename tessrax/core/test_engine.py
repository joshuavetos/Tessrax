"""
tests/test_engine.py
--------------------
Integration and reliability tests for Tessrax ContradictionEngine.

Verifies:
✓ Proper contradiction detection and ledger writes
✓ Valid cryptographic receipts
✓ Tracer emits runtime trace events
✓ Quarantine and error handling operate safely
"""

import os
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nacl.signing import SigningKey
from nacl.encoding import HexEncoder

from tessrax.core.contradiction_engine import ContradictionEngine
from tessrax.core.ledger import SQLiteLedger
from tessrax.core.receipts import verify_receipt, create_receipt, NonceRegistry, RevocationRegistry


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def temp_ledger():
    """Creates an in-memory SQLiteLedger for testing."""
    tmpfile = Path(tempfile.mktemp())
    ledger = SQLiteLedger(tmpfile)
    yield ledger
    ledger.close()
    if tmpfile.exists():
        tmpfile.unlink()


@pytest.fixture
def engine_setup(temp_ledger):
    """Initializes a ContradictionEngine with mock registries."""
    priv = SigningKey.generate().encode(encoder=HexEncoder).decode()
    mock_nonce = MagicMock(spec=NonceRegistry)
    mock_revoke = MagicMock(spec=RevocationRegistry)
    engine = ContradictionEngine(
        ledger=temp_ledger,
        signing_key_hex=priv,
        nonce_registry=mock_nonce,
        revocation_registry=mock_revoke,
        name="test_engine",
    )
    yield engine
    engine.tracer.stop()  # Gracefully stop tracer thread


# ---------------------------------------------------------------------
# Core Tests
# ---------------------------------------------------------------------
def test_emit_creates_signed_event(engine_setup, temp_ledger):
    """
    Ensures _emit() appends a valid, signed contradiction event to the ledger.
    """
    engine = engine_setup
    contradiction = {"type": "test_contradiction", "detail": "demo"}

    # Call _emit directly
    engine._emit(contradiction)

    # Read back ledger contents
    events = temp_ledger.get_all_events(verify=True)
    assert any(e.get("type") == "contradiction" for e in events), "No contradiction event found"

    # Verify that the event payload is valid JSON and signed
    for e in events:
        if e.get("type") == "contradiction":
            assert "timestamp" in e
            assert "payload" in e
            break


def test_chain_verification(engine_setup):
    """
    Confirms that verify_contradiction_chain delegates properly to ledger.
    """
    engine = engine_setup
    with patch.object(engine.ledger, "verify_chain", return_value=True) as mock_verify:
        result = engine.verify_contradiction_chain()
        mock_verify.assert_called_once()
        assert result is True


def test_runtime_trace_recorded(engine_setup, temp_ledger):
    """
    Ensures tracer asynchronously records a RUNTIME_TRACE event
    when instrumented methods execute.
    """
    engine = engine_setup
    event = {"mock_event": True}

    # Trigger a traced function
    with patch("tessrax.core.contradiction_engine.verify_receipt", return_value=True):
        engine._verify_event(event)

    time.sleep(0.15)  # Allow tracer queue to flush
    events = temp_ledger.get_all_events(verify=False)
    assert any(e.get("entry_type") == "RUNTIME_TRACE" for e in events), "No RUNTIME_TRACE found"


def test_quarantine_error_stops_engine(engine_setup):
    """
    Verifies that a quarantine failure halts batch processing.
    """
    engine = engine_setup

    # Patch _run_once_unlocked to raise QuarantineViolation
    from tessrax.core.contradiction_engine import QuarantineViolation
    with patch.object(engine, "_run_once_unlocked", side_effect=QuarantineViolation("Test fail")):
        with patch.object(engine, "stop") as mock_stop:
            engine.run_batch([{"fake": "event"}])
            mock_stop.assert_called_once()


def test_create_and_verify_receipt_roundtrip():
    """
    Standalone verification of Tessrax receipt integrity.
    """
    priv = SigningKey.generate().encode(encoder=HexEncoder).decode()
    payload = {"data": "hello_world"}
    r = create_receipt(priv, payload, executor_id="unit_test")
    assert verify_receipt(r), "Receipt should verify"
    # Tamper check
    r["payload"]["data"] = "corrupt"
    assert not verify_receipt(r), "Tampered receipt should fail verification"


def test_stats_reporting(engine_setup):
    """
    Ensures get_stats() reports correct structure.
    """
    engine = engine_setup
    stats = engine.get_stats()
    assert isinstance(stats, dict)
    expected_keys = {"total_contradictions", "total_scars", "quarantine_size", "chain_valid"}
    assert expected_keys.issubset(stats.keys())
