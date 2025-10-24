"""Schema definitions for Tessrax metabolic reconciliation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

try:  # pragma: no cover - optional dependency
    from pydantic import BaseModel, ConfigDict, Field, field_validator
except ModuleNotFoundError:  # pragma: no cover - fallback path
    BaseModel = None  # type: ignore[assignment]


if BaseModel is not None:

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

else:
    from dataclasses import dataclass, field

    @dataclass(frozen=True)
    class ClarityStatement:
        """Fallback implementation when Pydantic is unavailable."""

        event_type: Literal["CLARITY_GENERATION"] = "CLARITY_GENERATION"
        timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
        subject: str = ""
        metric: str = ""
        contradiction_ids: List[str] = field(default_factory=list)
        synthesis: str = ""
        confidence: float = 0.0
        action: str = "SYNTHESIZE_CLARITY"
        severity: str = "medium"
        clarity_fuel: float = 0.0
        rationale: str = ""
        audit_reference: Optional[str] = None

        def __post_init__(self) -> None:
            if self.timestamp.tzinfo is None:
                object.__setattr__(self, "timestamp", self.timestamp.replace(tzinfo=timezone.utc))
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError("confidence must be between 0.0 and 1.0")
            if self.clarity_fuel < 0.0:
                raise ValueError("clarity_fuel must be non-negative")

        def to_receipt(self) -> dict:
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
