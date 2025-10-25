"""Tests for the Truth-Lock FastAPI service primitives."""

from truth_lock_api import RedTeamSummary, TruthLockService


def test_query_returns_verified_answer_for_known_fact(tmp_path, monkeypatch):
    service = TruthLockService()

    # Ensure the ledger writes into a temporary file to keep the test isolated.
    monkeypatch.setattr("ledger_persistence.LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr("truth_lock_api.LEDGER_PATH", tmp_path / "ledger.jsonl")

    response = service.answer_query("What is the capital of France?")
    assert response.status == "verified"
    assert response.results[0].answer == "Paris"


def test_query_returns_unknown_for_unverified_fact(tmp_path, monkeypatch):
    service = TruthLockService()
    monkeypatch.setattr("ledger_persistence.LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr("truth_lock_api.LEDGER_PATH", tmp_path / "ledger.jsonl")

    response = service.answer_query("What is the capital of Atlantis?")
    assert response.status == "unknown"
    assert response.results == []


def test_red_team_suite_passes(tmp_path, monkeypatch):
    service = TruthLockService()
    monkeypatch.setattr("ledger_persistence.LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr("truth_lock_api.LEDGER_PATH", tmp_path / "ledger.jsonl")

    summary: RedTeamSummary = service.red_team_registry.run_falsifiability_suite(service)
    assert summary.all_passed is True
    assert all(result.passed for result in summary.results)
