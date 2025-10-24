"""FastAPI server for Tessrax-Core."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config_loader import load_config
from tessrax.tessrax_engine import calculate_stability, route_to_governance_lane

config = load_config()
app = FastAPI(title="Tessrax-Core API", version="1.0.0")

ledger_path = Path(config.logging.ledger_path)
ledger_path.parent.mkdir(parents=True, exist_ok=True)


class AgentClaim(BaseModel):
    agent: str = Field(..., description="Agent identifier")
    claim: str = Field(..., description="Agent claim text")
    context: Dict[str, Any] | None = Field(default=None, description="Optional claim metadata")


class ClaimsSubmission(BaseModel):
    claims: List[AgentClaim] = Field(..., description="Claims to analyse")


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

    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    return AnalysisResult(stability_score=stability, governance_lane=lane)


@app.get("/ledger")
async def get_ledger() -> JSONResponse:
    if not ledger_path.exists():
        return JSONResponse(content=[], status_code=200)

    records: List[Dict[str, Any]] = []
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - protect against corruption
                raise HTTPException(status_code=500, detail=f"Invalid ledger entry: {exc}")
    return JSONResponse(content=records)


@app.get("/healthz")
async def health() -> Dict[str, str]:
    return {"status": "ok"}
