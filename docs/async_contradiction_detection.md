# Tessrax Async Contradiction Detection

Safepoint Target: SAFEPOINT_ASYNC_V15_3b  
Version Target: v1.2.1  

## What This Module Does
The async contradiction detector replaces batch scanning with an event-driven pipeline.
When a new claim is received:
1. It is queued (subject to backpressure).
2. The detector looks up relevant historical claims (same subject / metric) via a SQLite index of the ledger.
3. The detector runs the contradiction engine on just that localized set.
4. Detected contradictions are written into the append-only ledger as governance decisions.

## Runtime Configuration
| Variable                  | Default                 | Description |
|--------------------------|-------------------------|-------------|
| `TESSRAX_ASYNC_WORKERS`  | `8`                     | Number of concurrent worker tasks |
| `LEDGER_PATH`            | `./data/ledger.jsonl`   | Append-only ledger file path |
| `QUEUE_MAXSIZE`          | `1000`                  | Max pending claims before drop |

## Observability
The FastAPI app exposes `GET /metrics`, which returns a JSON snapshot like:
```json
{
  "queue_depth": 0,
  "claims_seen": 120,
  "workers": 8,
  "published": 123,
  "dropped": 4,
  "detected": 119
}
```

Interpretation:
- queue_depth should remain low (<100 under normal load).
- dropped > 0 means backpressure is occurring (raise workers or investigate input rate).
- detected tracks how many governance decisions were appended.

## Shutdown Semantics

On shutdown:
1. The API stops accepting work.
2. The detector drains its queue (`queue.join()`).
3. Any remaining backlog is logged at WARN with depth.
4. Worker tasks and the periodic sync task are cancelled.
5. The SQLite index connection is closed cleanly.

## Test Targets (Must Pass Before Merge)
- Backpressure behavior: overflowing the queue increments dropped.
- Deduplication: posting the same claim twice only processes it once.
- Historical contradiction: two conflicting claims for the same subject/metric result in a governance receipt in the ledger.

## Governance Notes
- Epistemic integrity, drift, and severity scores reported in ledger entries are targets until verified in CI.
- SAFEPOINT tags (e.g. SAFEPOINT_ASYNC_V15_3b) must only be pushed after tests pass in CI.
- Do not claim “compliant”, “fault-tolerant”, or “stable” in public artifacts unless CI output is attached as proof.

This README describes intended behavior, configuration, and validation requirements. It does not self-certify outcomes.
