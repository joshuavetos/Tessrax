"""Schema definitions for Tessrax metabolic reconciliation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClarityStatement(BaseModel):
    """Canonical record describing a reconciliation event."""

    event_type: Literal["CLARITY_GENERATION"] = Field(default="CLARITY_GENERATION")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    subject: str
    metric: str
    contradiction_ids: List[str]
    synthesis: str
    confidence: float = Field(ge=0.0, le=1.0)
    action: str = Field(default="SYNTHESIZE_CLARITY")
    severity: str
    clarity_fuel: float = Field(default=0.0, ge=0.0)
    rationale: str
    audit_reference: Optional[str] = None

    @field_validator("timestamp", mode="before")
    def _ensure_timezone(cls, value: datetime) -> datetime:
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    def to_receipt(self) -> dict:
        """Return a JSON-serialisable ledger payload."""

        payload = {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "subject": self.subject,
            "metric": self.metric,
            "contradiction_ids": list(self.contradiction_ids),
            "confidence": round(self.confidence, 3),
            "synthesis": self.synthesis,
            "action": self.action,
            "severity": self.severity,
            "clarity_fuel": round(self.clarity_fuel, 3),
            "rationale": self.rationale,
        }
        if self.audit_reference:
            payload["audit_reference"] = self.audit_reference
        return payload

    model_config = ConfigDict(frozen=True)
