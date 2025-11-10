from tessrax.core.ledger import Ledger


def test_verify_accepts_ledger_with_runtime_timestamp(tmp_path, monkeypatch) -> None:
    """Entries written by :class:`Ledger` include a runtime timestamp."""

    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    ledger_path = ledger_dir / "ledger.jsonl"

    ledger = Ledger(path=ledger_path)
    ledger.append({"event": "alpha"})

    monkeypatch.chdir(tmp_path)

    from core import verify_ledger

    assert verify_ledger.verify()
