"""Tests for token minting and license embedding contracts."""

from tessrax.core.contracts.token import TokenBank
from tessrax.core.contracts.license import embed_license
import json


def test_token_mint_and_balance(tmp_path):
    bank = TokenBank(tmp_path / "rewards.jsonl")
    e1 = bank.mint("node_A", 5, "federated_quorum")
    assert e1["amount"] == 5
    assert bank.balance("node_A") == 5


def test_license_embed(tmp_path):
    event = {"event": "quorum_pass", "actor": "node_A"}
    wrapped = embed_license(event, "Test-License-0.1")
    assert "license" in wrapped
    data = json.loads(json.dumps(wrapped))
    assert data["payload"]["event"] == "quorum_pass"
