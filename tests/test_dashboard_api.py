from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from dashboard.api_routes import create_dashboard_router


def _write_ledger(path: Path) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stability_score": 0.9,
        "governance_lane": "autonomic",
        "claims": [{"agent": "a", "claim": "alpha"}],
    }
    path.write_text(f"{payload}\n".replace("'", '"'), encoding="utf-8")


def test_dashboard_routes(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger)

    def metrics_provider() -> dict[str, float]:
        return {"epistemic_integrity": 0.95, "epistemic_drift": 0.02}

    app = FastAPI()
    app.include_router(
        create_dashboard_router(
            ledger_path=ledger,
            thresholds={"autonomic": 0.8, "deliberative": 0.5},
            metrics_provider=metrics_provider,
        )
    )

    client = TestClient(app)
    summary = client.get("/dashboard/api/summary")
    assert summary.status_code == 200
    assert summary.json()["total"] == 1

    ledger_resp = client.get("/dashboard/api/ledger")
    assert ledger_resp.status_code == 200
    assert len(ledger_resp.json()["records"]) == 1

    metrics_resp = client.get("/dashboard/api/metrics")
    assert metrics_resp.status_code == 200
    assert metrics_resp.json()["epistemic_integrity"] == 0.95
