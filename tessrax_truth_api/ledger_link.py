"""Decorators for ledger aware endpoints."""
from __future__ import annotations

import datetime
import inspect
from functools import wraps
from typing import Any, Callable, Dict

from fastapi import Request

from tessrax_truth_api.services.provenance_service import ProvenanceService


def ledger_guard(provenance: ProvenanceService, event_type: str) -> Callable:
    """Decorate an endpoint to capture audit events in the ledger."""

    def decorator(func: Callable) -> Callable:
        is_async = inspect.iscoroutinefunction(func)

        if is_async:

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                response = await func(*args, **kwargs)
                _record_event(provenance, event_type, args, kwargs, response)
                return response

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            response = func(*args, **kwargs)
            _record_event(provenance, event_type, args, kwargs, response)
            return response

        return sync_wrapper

    return decorator


def _record_event(
    provenance: ProvenanceService,
    event_type: str,
    args: tuple[Any, ...],
    kwargs: Dict[str, Any],
    response: Any,
) -> None:
    payload = {
        "event": event_type,
        "arguments": _serialise_args(args, kwargs),
        "response_summary": _summarise_response(response),
    }
    provenance.record_event(event_type, payload)


def _serialise_args(args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    data: Dict[str, Any] = {f"arg_{idx}": _maybe_model_dump(value) for idx, value in enumerate(args)}
    data.update({key: _maybe_model_dump(value) for key, value in kwargs.items()})
    return data


def _summarise_response(response: Any) -> Any:
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump(mode="json")
        except TypeError:  # pragma: no cover - fallback for incompatible signatures
            return response.model_dump()
    if isinstance(response, dict):
        return response
    return str(response)


def _maybe_model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, Request):  # pragma: no cover - request bodies not serialised
        return {"path": value.url.path}
    if isinstance(value, (tuple, list)):
        return [_maybe_model_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _maybe_model_dump(val) for key, val in value.items()}
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
