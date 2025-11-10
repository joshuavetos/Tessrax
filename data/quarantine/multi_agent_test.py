"""Run a concurrency-backed consistency check across governance agents."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from config_loader import load_config
from tessrax.contradiction import ContradictionEngine
from tessrax.types import Claim, ContradictionRecord

LOG_PATH = Path("tests/logs/multi_agent_consistency.log")
AGENTS: Sequence[str] = ("alpha", "beta", "gamma", "delta", "epsilon")


def _base_timestamp() -> datetime:
    return datetime.now(timezone.utc)


def _noise(agent_id: str) -> random.Random:
    seed = sum(ord(char) for char in agent_id)
    return random.Random(seed)


def _generate_claims(agent_id: str) -> list[Claim]:
    rng = _noise(agent_id)
    timestamp = _base_timestamp()
    claims: list[Claim] = []
    subjects = (
        ("grid", "availability", 0.91, 0.81),
        ("water", "purity", 0.97, 0.86),
        ("safety", "incident_rate", 0.04, 0.17),
    )
    for subject, metric, baseline, challenger in subjects:
        jitter = rng.uniform(-0.005, 0.005)
        challenger_jitter = rng.uniform(-0.008, 0.008)
        claims.append(
            Claim(
                claim_id=f"{agent_id}-{subject}-ops",
                subject=subject,
                metric=metric,
                value=round(baseline + jitter, 4),
                unit="ratio",
                timestamp=timestamp,
                source=f"ops::{agent_id}",
            )
        )
        claims.append(
            Claim(
                claim_id=f"{agent_id}-{subject}-audit",
                subject=subject,
                metric=metric,
                value=round(challenger + challenger_jitter, 4),
                unit="ratio",
                timestamp=timestamp,
                source=f"audit::{agent_id}",
            )
        )
    return claims


def _summarise(contradictions: Iterable[ContradictionRecord]) -> str:
    tokens = {
        f"{item.claim_a.subject}:{item.claim_a.metric}" for item in contradictions
    }
    return " | ".join(sorted(tokens))


def _semantic_overlap(first: str, second: str) -> float:
    if not first or not second:
        return 0.0
    return SequenceMatcher(None, first, second).ratio()


AgentReport = dict[str, object]


def _evaluate_agent(agent_id: str, tolerance: float) -> AgentReport:
    engine = ContradictionEngine(tolerance=tolerance)
    claims = _generate_claims(agent_id)
    contradictions = engine.detect(claims)
    summary = _summarise(contradictions)
    return {
        "agent": agent_id,
        "contradictions": [record.to_json() for record in contradictions],
        "summary": summary,
    }


def _compute_consistency(summaries: Sequence[str]) -> float:
    if len(summaries) < 2:
        return 1.0
    overlaps: list[float] = []
    for idx in range(len(summaries)):
        for jdx in range(idx + 1, len(summaries)):
            overlaps.append(_semantic_overlap(summaries[idx], summaries[jdx]))
    if not overlaps:
        return 0.0
    return sum(overlaps) / len(overlaps)


def _write_log(payload: AgentReport) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")


def run() -> None:
    config = load_config()
    tolerance = max(0.02, 0.15 * (1.0 - config.thresholds.get("deliberative", 0.5)))

    with ThreadPoolExecutor(max_workers=len(AGENTS)) as executor:
        futures = {
            executor.submit(_evaluate_agent, agent, tolerance): agent
            for agent in AGENTS
        }
        results: list[AgentReport] = []
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: str(item["agent"]))

    summaries: list[str] = [str(item.get("summary", "")) for item in results]
    consistency = _compute_consistency(summaries)
    passed = consistency >= 0.85

    log_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "consistency_score": round(consistency, 4),
        "passed": passed,
        "agents": results,
        "tolerance": tolerance,
    }
    _write_log(log_payload)

    print("Multi-agent consistency evaluation complete.")
    print(f"Semantic overlap score: {consistency:.4f}")
    print(f"Pass threshold: 0.85 → {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(
            "Consistency check failed; inspect tests/logs/multi_agent_consistency.log"
        )


if __name__ == "__main__":
    run()
