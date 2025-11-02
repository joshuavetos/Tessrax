"""FastAPI server for Tessrax-Core."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config_loader import load_config
from dashboard.api_routes import create_dashboard_router
from tessrax.ledger import Ledger
from tessrax.logging import LedgerWriter, S3LedgerWriter
from tessrax.metabolism.async_detector import AsyncContradictionDetector
from tessrax.tessrax_engine import calculate_stability, route_to_governance_lane
from tessrax.types import Claim

config = load_config()

ledger_path = Path(config.logging.ledger_path)
cloud_writer = None
if config.logging.cloud and config.logging.cloud.enabled:
    cloud_writer = S3LedgerWriter(config.logging.cloud)
ledger_writer = LedgerWriter(ledger_path, cloud_writer)

governance_ledger = Ledger()
detector = AsyncContradictionDetector(governance_ledger)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await detector.start()
    try:
        yield
    finally:
        await detector.stop()
        ledger_writer.close()


app = FastAPI(title="Tessrax-Core API", version="1.0.0", lifespan=lifespan)


class AgentClaim(BaseModel):
    agent: str = Field(..., description="Agent identifier")
    claim: str = Field(..., description="Agent claim text")
    context: dict[str, Any] | None = Field(
        default=None, description="Optional claim metadata"
    )


class ClaimsSubmission(BaseModel):
    claims: list[AgentClaim] = Field(..., description="Claims to analyse")


class AnalysisResult(BaseModel):
    stability_score: float = Field(..., ge=0.0, le=1.0)
    governance_lane: str


@app.post("/submit_claims", response_model=AnalysisResult)
async def submit_claims(submission: ClaimsSubmission) -> AnalysisResult:
    if not submission.claims:
        raise HTTPException(status_code=400, detail="At least one claim is required")

    raw_claims = [claim.dict() for claim in submission.claims]
    stability = calculate_stability(raw_claims)
    lane = route_to_governance_lane(stability, config.thresholds)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stability_score": stability,
        "governance_lane": lane,
        "claims": raw_claims,
    }

    ledger_writer.append(record)

    return AnalysisResult(stability_score=stability, governance_lane=lane)


@app.get("/ledger")
async def get_ledger() -> JSONResponse:
    if not ledger_path.exists():
        return JSONResponse(content=[], status_code=200)

    records: list[dict[str, Any]] = []
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except (
                json.JSONDecodeError
            ) as exc:  # pragma: no cover - protect against corruption
                raise HTTPException(
                    status_code=500, detail=f"Invalid ledger entry: {exc}"
                )
    return JSONResponse(content=records)


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


class StructuredClaim(BaseModel):
    """Pydantic payload used for asynchronous contradiction detection."""

    claim_id: str
    subject: str
    metric: str
    value: float
    unit: str
    timestamp: datetime
    source: str
    context: dict[str, str] = Field(default_factory=dict)

    def to_claim(self) -> Claim:
        return Claim(
            claim_id=self.claim_id,
            subject=self.subject,
            metric=self.metric,
            value=self.value,
            unit=self.unit,
            timestamp=self.timestamp,
            source=self.source,
            context=dict(self.context),
        )


@app.post("/claims")
async def submit_structured_claims(claims: list[StructuredClaim]) -> dict[str, object]:
    if not claims:
        raise HTTPException(status_code=400, detail="At least one claim is required")

    for payload in claims:
        await detector.publish(payload.to_claim())
    return {"status": "queued", "count": len(claims)}


@app.get("/contradictions/live")
async def get_live_contradictions() -> dict[str, int]:
    return {"active": len(detector.seen)}


@app.get("/metrics")
async def metrics() -> dict[str, int]:
    return detector.metrics()


dashboard_static = (
    Path(__file__).resolve().parent.parent / "dashboard" / "react_app"
)
if dashboard_static.exists():
    app.mount(
        "/dashboard",
        StaticFiles(directory=dashboard_static, html=True),
        name="dashboard",
    )


@app.get("/", response_class=HTMLResponse)
async def root_dashboard() -> HTMLResponse:
    if dashboard_static.exists():
        index_file = dashboard_static / "index.html"
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Tessrax Governance API</h1>")


app.include_router(
    create_dashboard_router(
        ledger_path=ledger_path,
        thresholds=config.thresholds,
        metrics_provider=detector.metrics,
    )
)
