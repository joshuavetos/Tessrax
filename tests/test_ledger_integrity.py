"""Additional integrity checks for :mod:`tessrax.ledger`."""

from tessrax.ledger import Ledger
from tests.test_ledger import make_decision


def test_ledger_hash_continuity() -> None:
    ledger = Ledger()
    first_receipt = ledger.append(make_decision())
    second_receipt = ledger.append(make_decision())

    assert second_receipt.prev_hash == first_receipt.hash
