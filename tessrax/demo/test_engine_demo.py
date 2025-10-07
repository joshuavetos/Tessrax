"""
Automated test harness for the Tessrax ContradictionEngine demo.
Validates core engine behavior using the mock dependencies.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Ensure imports work whether this is run directly or via pytest
sys.path.append(str(Path(__file__).resolve().parent))

from tessrax.core.contradiction_engine_demo import (
    ContradictionEngine,
    ILedger,
    NonceRegistry,
    RevocationRegistry,
    verify_receipt,
    Tracer,
    trace,
    ResourceMonitor,
    ensure_in_sandbox,
    example_rule_value_mismatch,
    example_metabolize_contradiction,
)

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_engine_demo")


def run_engine_test():
    """Run a basic validation of ContradictionEngine using mock components."""
    logger.info("Starting ContradictionEngine demo test...")

    dummy_key_hex = "00" * 32

    # Instantiate mock dependencies
    ledger = ILedger("test_ledger.db")
    nonce_registry = NonceRegistry()
    revocation_registry = RevocationRegistry()

    engine = ContradictionEngine(
        ledger=ledger,
        ruleset=[example_rule_value_mismatch],
        signing_key_hex=dummy_key_hex,
        nonce_registry=nonce_registry,
        revocation_registry=revocation_registry,
        name="engine_test",
        verify_strict=False,
        quarantine_path="data/quarantine/test_quarantine.jsonl",
        metabolize_fn=example_metabolize_contradiction,
    )

    # Create events
    events = [
        {"id": "event_match", "inputs": {"input_value": 10}, "outputs": {"output_value": 10}, "receipt": {"dummy_receipt_data": "ok"}},
        {"id": "event_contradiction", "inputs": {"input_value": 5}, "outputs": {"output_value": 9}, "receipt": {"dummy_receipt_data": "ok"}},
        {"id": "event_noreceipt", "inputs": {"input_value": 1}, "outputs": {"output_value": 2}},
    ]

    # Run batch
    engine.run_batch(events)

    # Gather stats
    stats = engine.get_stats()
    logger.info("Engine stats:\n%s", json.dumps(stats, indent=2))

    # Assertions
    assert "total_contradictions" in stats, "Missing total_contradictions field."
    assert stats["total_contradictions"] >= 1, "Expected at least one contradiction."
    assert stats["chain_valid"] is True, "Ledger chain should verify as valid."

    print("\n✅ ContradictionEngine test passed successfully.")
    print(json.dumps(stats, indent=2))

    # Cleanup artifacts
    if engine._quarantine_path.exists():
        os.remove(engine._quarantine_path)
        logger.info("Cleaned up quarantine log.")
    if Path("test_ledger.db").exists():
        os.remove("test_ledger.db")
        logger.info("Cleaned up ledger DB.")


if __name__ == "__main__":
    run_engine_test()
