from __future__ import annotations

import json

from tessrax.analytics.pattern_extractor import extract_patterns_from_stream
from tessrax.metabolism.adversarial import AdversarialAgent


def test_adversarial_batch_size() -> None:
    agent = AdversarialAgent(seed=1, max_batch=5)
    batch = agent.run_batch(4)
    assert len(batch) == 4
    assert all(r["event_type"] == "ADVERSARIAL_CONTRADICTION" for r in batch)


def test_pattern_extractor() -> None:
    lines = [
        json.dumps({"contradiction_type": "semantic", "resolution": {"result": "merge"}}),
        json.dumps({"contradiction_type": "procedural", "resolution": {"result": "fix"}}),
    ]
    pat = extract_patterns_from_stream(lines)
    assert pat["semantic"]["merge"] == 1.0
    assert pat["procedural"]["fix"] == 1.0
