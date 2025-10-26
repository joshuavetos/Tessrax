"""Pydantic data transfer objects for the Truth API."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    claim_a: str = Field(..., description="First claim to compare")
    claim_b: str = Field(..., description="Second claim to compare")
    tier: str = Field(..., description="Billing tier making the request")


class DetectResponse(BaseModel):
    score: float
    verdict: str
    status: Literal["verified", "unknown"]
    receipt_uuid: str
    timestamp: datetime
    signature: str


class ReceiptVerification(BaseModel):
    uuid: str
    status: Literal["verified", "unknown"]
    payload: dict
    merkle_hash: str
    signature_valid: bool


class OnboardResponse(BaseModel):
    tier: str
    token: str
    expires_in_minutes: int


class HealthResponse(BaseModel):
    status: str
    evaluated_at: datetime
    integrity: float
    drift: float
    severity: float


class SelfTestResult(BaseModel):
    name: str
    status: Literal["verified", "unknown", "tampered"]
    receipt_uuid: Optional[str]
    details: str


class SelfTestSummary(BaseModel):
    results: list[SelfTestResult]
    ledger_path: str


DetectRequest.model_rebuild()
DetectResponse.model_rebuild()
ReceiptVerification.model_rebuild()
OnboardResponse.model_rebuild()
HealthResponse.model_rebuild()
SelfTestResult.model_rebuild()
SelfTestSummary.model_rebuild()
