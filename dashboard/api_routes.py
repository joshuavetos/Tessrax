"""FastAPI routes exposing ledger telemetry for the React dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query

from tessrax.tessrax_engine import calculate_stability, route_to_governance_lane


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise HTTPException(status_code=500, detail=f"Invalid ledger entry: {exc}")
    return records


def _enrich(records: list[dict[str, Any]], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        claims = record.get("claims", [])
        stability = calculate_stability(claims)
        lane = route_to_governance_lane(stability, thresholds)
        enriched.append({**record, "stability_score": stability, "governance_lane": lane})
    return enriched


def create_dashboard_router(
    ledger_path: Path,
    thresholds: dict[str, float],
    metrics_provider: Callable[[], dict[str, Any]],
) -> APIRouter:
    router = APIRouter(prefix="/dashboard/api", tags=["dashboard"])

    @router.get("/ledger")
    async def ledger(limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
        records = _enrich(_load_records(ledger_path), thresholds)
        return {"records": records[-limit:]}

    @router.get("/summary")
    async def summary() -> dict[str, Any]:
        records = _enrich(_load_records(ledger_path), thresholds)
        lanes: dict[str, int] = {}
        for record in records:
            lane = record.get("governance_lane", "unknown")
            lanes[lane] = lanes.get(lane, 0) + 1
        return {"total": len(records), "lanes": lanes}

    @router.get("/metrics")
    async def metrics() -> dict[str, Any]:
        metrics_data = metrics_provider()
        if not isinstance(metrics_data, dict):
            raise HTTPException(status_code=500, detail="Metrics provider returned invalid data")
        return metrics_data

    return router


__all__ = ["create_dashboard_router"]
