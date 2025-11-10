"""Compatibility shim and façade for the Tessrax contradiction engine.

This module exposes a simple ``detect_contradictions`` helper that accepts
plain Python ``dict`` instances as used by the runtime behaviour validation
scripts.  The helper converts those mappings into
:class:`~tessrax.contradiction.Claim` objects, delegates contradiction detection
to the canonical :class:`~tessrax.contradiction.ContradictionEngine`, and
returns auditable payloads that respect the Tessrax governance clauses. Runtime
verification is performed on every inbound claim to satisfy the Codex Iron-Law
directives.

All emitted artefacts include an explicit governance signature block:
``{"auditor": "Tessrax Governance Kernel v16", "clauses": [...]}`` and a
``"status": "DLK-VERIFIED"`` tag so downstream tooling can assert provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping, MutableMapping, Sequence

from tessrax.contradiction import Claim, ContradictionEngine
from tessrax.governance import ContradictionRecord

__all__ = ["ContradictionEngine", "detect_contradictions", "last_contradictions"]


@dataclass(slots=True)
class _NormalisedClaim:
    """Internal helper ensuring the claim is serialisation ready."""

    claim: Claim
    raw: Mapping[str, object]


_CLAIM_UNIT_FALLBACK = "unitless"
_CLAIM_SOURCE_FALLBACK = "runtime_metabolic_loop"
_CLAIM_CONTEXT_KEY = "context"

_engine_singleton = ContradictionEngine()
_last_contradictions: list[ContradictionRecord] = []


def _ensure_timestamp(value: object) -> datetime:
    """Coerce timestamps into timezone-aware :class:`datetime` instances.

    Raises
    ------
    ValueError
        If the timestamp cannot be interpreted.
    """

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Invalid ISO timestamp: {value!r}") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    raise ValueError(f"Unsupported timestamp type: {type(value)!r}")


def _coerce_claim(index: int, payload: Mapping[str, object]) -> Claim:
    """Convert a loose mapping into a :class:`Claim` with validation."""

    required_fields = {"subject", "metric", "value", "timestamp"}
    missing = sorted(field for field in required_fields if field not in payload)
    if missing:
        raise ValueError(f"Claim {index} missing fields: {', '.join(missing)}")

    claim_id = payload.get("claim_id") or f"auto-claim-{index}"
    subject = str(payload["subject"]).strip()
    metric = str(payload["metric"]).strip()
    value = float(payload["value"])
    timestamp = _ensure_timestamp(payload["timestamp"])
    unit = str(payload.get("unit", _CLAIM_UNIT_FALLBACK) or _CLAIM_UNIT_FALLBACK)
    source = str(payload.get("source", _CLAIM_SOURCE_FALLBACK) or _CLAIM_SOURCE_FALLBACK)
    raw_context = payload.get(_CLAIM_CONTEXT_KEY, {})
    context = dict(raw_context) if isinstance(raw_context, Mapping) else {}

    if not subject:
        raise ValueError(f"Claim {index} subject is empty")
    if not metric:
        raise ValueError(f"Claim {index} metric is empty")

    return Claim(
        claim_id=str(claim_id),
        subject=subject,
        metric=metric,
        value=value,
        unit=unit,
        timestamp=timestamp,
        source=source,
        context=context,
    )


def _normalise_claims(claims: Iterable[Mapping[str, object]]) -> list[_NormalisedClaim]:
    """Validate and convert inbound claims into :class:`Claim` instances."""

    normalised: list[_NormalisedClaim] = []
    for index, raw_claim in enumerate(claims):
        if isinstance(raw_claim, Claim):
            claim_obj = raw_claim
            raw_mapping: MutableMapping[str, object] = {
                "claim_id": claim_obj.claim_id,
                "subject": claim_obj.subject,
                "metric": claim_obj.metric,
                "value": claim_obj.value,
                "unit": claim_obj.unit,
                "timestamp": claim_obj.timestamp.isoformat(),
                "source": claim_obj.source,
            }
            if claim_obj.context:
                raw_mapping[_CLAIM_CONTEXT_KEY] = dict(claim_obj.context)
        elif isinstance(raw_claim, Mapping):
            claim_obj = _coerce_claim(index, raw_claim)
            raw_mapping = dict(raw_claim)
        else:  # pragma: no cover - defensive branch
            raise TypeError(
                "Claims must be Mapping or Claim instances; "
                f"received {type(raw_claim)!r}"
            )
        normalised.append(_NormalisedClaim(claim=claim_obj, raw=raw_mapping))
    if not normalised:
        raise ValueError("At least one claim is required for contradiction detection")
    return normalised


def detect_contradictions(
    claims: Sequence[Mapping[str, object] | Claim], *, engine: ContradictionEngine | None = None
) -> list[dict[str, object]]:
    """Detect contradictions and emit DLK-verified payloads.

    Parameters
    ----------
    claims:
        Sequence of claim mappings compatible with the runtime metabolic loop
        test.  Each mapping MUST include ``subject``, ``metric``, ``value``, and
        ``timestamp``.
    engine:
        Optional explicit :class:`ContradictionEngine` instance.  Defaults to the
        module singleton to preserve energy accounting.

    Returns
    -------
    list of dict
        Serialised contradiction payloads including governance metadata.
    """

    active_engine = engine or _engine_singleton
    normalised = _normalise_claims(claims)
    records = active_engine.detect([item.claim for item in normalised])
    global _last_contradictions
    _last_contradictions = list(records)

    serialised: list[dict[str, object]] = []
    for record in records:
        payload = record.to_json()
        payload.update(
            {
                "auditor": "Tessrax Governance Kernel v16",
                "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
                "status": "DLK-VERIFIED",
                "runtime_info": {
                    "metabolic_delta": round(record.delta, 6),
                    "kappa": round(record.kappa, 6),
                    "energy": round(record.energy, 6),
                },
            }
        )
        serialised.append(payload)
    return serialised


def last_contradictions() -> list[ContradictionRecord]:
    """Expose the most recent contradiction records for downstream modules."""

    return list(_last_contradictions)

