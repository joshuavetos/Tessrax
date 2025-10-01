import json
import os
import pytest
from src.tessrax_engine.engine import TessraxEngine

def test_metabolize_generates_fuel():
    engine = TessraxEngine()
    contradictions = [("A", "B")]
    fuel = engine._metabolize(contradictions, "test")
    assert isinstance(fuel, list)
    assert "fuel" in fuel[0]

def test_sign_and_verify_claim():
    engine = TessraxEngine()
    claim = engine._sign_claim("tester", "This is a test")
    assert engine._verify_claim(claim)

def test_handoff_chain_integrity(tmp_path):
    handoff_file = tmp_path / "handoffs.jsonl"
    engine = TessraxEngine(handoff_file=str(handoff_file))
    contradictions = [("X", "Y")]
    output = engine.run(contradictions, "Testing")
    assert engine.verify_handoff_chain()
