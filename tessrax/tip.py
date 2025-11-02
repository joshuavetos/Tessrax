"""Tessrax Integration Protocol (TIP) handshake models."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

try:  # pragma: no cover - optional dependency path
    from pydantic import BaseModel, ConfigDict, Field, field_validator
except ModuleNotFoundError:  # pragma: no cover - fallback path
    BaseModel = None  # type: ignore[assignment]
    ConfigDict = Field = field_validator = None  # type: ignore[assignment]

TIP_PROTOCOL = "TIP"


def load_tip_schema() -> dict[str, Any]:
    """Load the JSON schema describing the Tessrax Integration Protocol."""

    schema_path = Path(__file__).resolve().parent.parent / "schemas" / "tip.schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"TIP schema not found: {schema_path}")
    import json

    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if BaseModel is not None:

    class TIPHandshake(BaseModel):
        """Structured TIP handshake envelope with validation hooks."""

        protocol: Literal[TIP_PROTOCOL] = Field(default=TIP_PROTOCOL)
        version: str = Field(pattern=r"^v\d+\.\d+$", default="v1.0")
        source: str
        target: str
        timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
        nonce: str
        capabilities: list[str]
        integrity: float = Field(ge=0.0, le=1.0, default=1.0)
        status: Literal["proposal", "accepted", "rejected"] = "proposal"
        signature: str
        metadata: dict[str, Any] = Field(default_factory=dict)

        model_config = ConfigDict(frozen=True)

        @field_validator("capabilities")
        @classmethod
        def _validate_capabilities(cls, value: list[str]) -> list[str]:
            if not value:
                raise ValueError("capabilities must contain at least one entry")
            deduplicated = list(dict.fromkeys(value))
            if len(deduplicated) != len(value):
                raise ValueError("capabilities must be unique")
            if any(not item for item in deduplicated):
                raise ValueError("capabilities entries must be non-empty")
            return deduplicated

        @field_validator("nonce")
        @classmethod
        def _validate_nonce(cls, value: str) -> str:
            if len(value) < 8:
                raise ValueError("nonce must be at least 8 characters long")
            return value

        @field_validator("signature")
        @classmethod
        def _validate_signature(cls, value: str) -> str:
            if len(value) < 32:
                raise ValueError("signature must be at least 32 characters long")
            return value

        @field_validator("timestamp", mode="before")
        @classmethod
        def _ensure_timezone(cls, value: datetime) -> datetime:
            if isinstance(value, datetime) and value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value

        def to_payload(self) -> dict[str, Any]:
            payload = self.model_dump(mode="json")
            payload["timestamp"] = self.timestamp.isoformat()
            return payload

else:

    from dataclasses import dataclass, field

    @dataclass(frozen=True)
    class TIPHandshake:
        """Fallback dataclass for environments without Pydantic."""

        protocol: Literal[TIP_PROTOCOL] = TIP_PROTOCOL
        version: str = "v1.0"
        source: str = ""
        target: str = ""
        timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
        nonce: str = ""
        capabilities: list[str] = field(default_factory=list)
        integrity: float = 1.0
        status: Literal["proposal", "accepted", "rejected"] = "proposal"
        signature: str = ""
        metadata: dict[str, Any] = field(default_factory=dict)

        def __post_init__(self) -> None:
            if not self.source or not self.target:
                raise ValueError("source and target must be provided")
            if len(self.nonce) < 8:
                raise ValueError("nonce must be at least 8 characters long")
            if len(self.signature) < 32:
                raise ValueError("signature must be at least 32 characters long")
            if not 0.0 <= self.integrity <= 1.0:
                raise ValueError("integrity must be between 0.0 and 1.0")
            if not self.capabilities:
                raise ValueError("capabilities must contain at least one entry")
            if len(set(self.capabilities)) != len(self.capabilities):
                raise ValueError("capabilities must be unique")
            if self.timestamp.tzinfo is None:
                object.__setattr__(self, "timestamp", self.timestamp.replace(tzinfo=timezone.utc))

        def to_payload(self) -> dict[str, Any]:
            return {
                "protocol": self.protocol,
                "version": self.version,
                "source": self.source,
                "target": self.target,
                "timestamp": self.timestamp.isoformat(),
                "nonce": self.nonce,
                "capabilities": list(self.capabilities),
                "integrity": self.integrity,
                "status": self.status,
                "signature": self.signature,
                "metadata": dict(self.metadata),
            }

    TIPHandshake = cast(type[TIPHandshake], TIPHandshake)


__all__ = ["TIPHandshake", "TIP_PROTOCOL", "load_tip_schema"]
