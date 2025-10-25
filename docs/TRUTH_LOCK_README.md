# Truth-Lock Prototype

The Truth-Lock prototype is a deterministic FastAPI microservice that answers a
small set of curated questions. Every interaction is recorded to an append-only
ledger so that auditors can trace responses over time. A red-team
``falsifiability`` suite is included to demonstrate how the system resists
hallucinated answers.

## Getting Started

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the API server

```bash
uvicorn truth_lock_api:app --host 0.0.0.0 --port 8000
```

The service exposes the following endpoints:

* `POST /query` – verify a natural language question.
* `GET /red-team/test` – execute the falsifiability suite.
* `GET /ledger/entries` – inspect the most recent ledger entries.
* `GET /health` – liveness information.

## Running the Tests

```bash
pytest -q
```

## Red-Team Falsifiability Suite

`TruthLockService.red_team_registry` includes tests that confirm:

* canonical facts (e.g. the capital of France) return `verified`.
* unknown facts are rejected with a `status` of `unknown`.

The suite is accessible through Python:

```python
from truth_lock_api import TruthLockService
svc = TruthLockService()
print(svc.red_team_registry.run_falsifiability_suite(svc))
```

## Governance Considerations

* **Zero-hallucination enforcement** – unrecognised questions do not return a
  best guess. They are marked as `unknown` so operators can improve coverage.
* **Ledger transparency** – every interaction appends to `ledger.jsonl`. The
  append-only model supports auditing and tamper detection.

## Example Query

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}' \
  http://localhost:8000/query
```
