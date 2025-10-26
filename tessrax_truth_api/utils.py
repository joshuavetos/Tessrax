"""Utility helpers for the Tessrax Truth API runtime."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import dotenv_values
import jwt

_CONFIG_LOCK = threading.Lock()
_ENV_LOCK = threading.Lock()


def _config_path() -> Path:
    return Path(__file__).resolve().parent / "config.yml"


def _env_path() -> Path:
    return Path(__file__).resolve().parent / ".env"


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """Load the YAML configuration for the API."""

    with _CONFIG_LOCK:
        with _config_path().open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)


@lru_cache(maxsize=1)
def load_env() -> Dict[str, str]:
    """Load secret values from the local ``.env`` file."""

    with _ENV_LOCK:
        env_path = _env_path()
        if env_path.exists():
            return {**dotenv_values(env_path), **os.environ}
        return dict(os.environ)


def hmac_signature(payload: Dict[str, Any], secret: Optional[str] = None) -> str:
    """Create a base64 encoded HMAC signature for the payload."""

    env = load_env()
    secret_value = secret or env.get("HMAC_SECRET") or load_config().get("hmac_secret")
    if not secret_value:
        raise RuntimeError("HMAC secret is not configured")
    digest = hmac.new(
        secret_value.encode("utf-8"),
        json.dumps(payload, sort_keys=True).encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8")


def verify_signature(payload: Dict[str, Any], signature: str, secret: Optional[str] = None) -> bool:
    """Verify a signature for a payload."""

    expected = hmac_signature(payload, secret=secret)
    return hmac.compare_digest(expected, signature)


def issue_jwt(tier: str, subject: str, *, minutes: Optional[int] = None) -> str:
    """Issue a short lived JWT for the supplied tier."""

    config = load_config()
    env = load_env()
    secret = env.get("JWT_SECRET") or config.get("jwt", {}).get("secret")
    if not secret:
        raise RuntimeError("JWT secret is not configured")

    expires_delta = minutes or config.get("jwt", {}).get("expires_minutes", 60)
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "tier": tier,
        "iss": config.get("jwt", {}).get("issuer", "tessrax-truth-api"),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_delta)).timestamp()),
    }
    token = jwt.encode(payload, secret, algorithm=config.get("jwt", {}).get("algorithm", "HS256"))
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token


def decode_jwt(token: str) -> Dict[str, Any]:
    """Decode and verify a JWT returning the payload."""

    config = load_config()
    env = load_env()
    secret = env.get("JWT_SECRET") or config.get("jwt", {}).get("secret")
    if not secret:
        raise RuntimeError("JWT secret is not configured")
    return jwt.decode(
        token,
        secret,
        algorithms=[config.get("jwt", {}).get("algorithm", "HS256")],
        issuer=config.get("jwt", {}).get("issuer", "tessrax-truth-api"),
    )


def utcnow() -> datetime:
    """Return a timezone aware ``datetime`` in UTC."""

    return datetime.now(timezone.utc)


def receipt_identifier(*, seed: Optional[str] = None) -> str:
    """Return a deterministic UUID5 when a seed is provided."""

    import uuid

    if seed is None:
        return str(uuid.uuid4())
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def merkle_hash(*, payload: Dict[str, Any], prev_hash: Optional[str]) -> str:
    """Compute a deterministic hash for ledger entries."""

    hasher = hashlib.sha256()
    hasher.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    if prev_hash:
        hasher.update(prev_hash.encode("utf-8"))
    return hasher.hexdigest()


def base_metrics_snapshot(calibration: Dict[str, Any]) -> Dict[str, float]:
    """Return governance metrics from calibration config."""

    return {
        "integrity": float(calibration.get("integrity_min", 0.94)),
        "drift": float(calibration.get("drift_max", 0.04)),
        "severity": float(calibration.get("severity_max", 0.05)),
    }


def encode_metrics(metrics: Dict[str, float]) -> str:
    """Encode metrics into a Prometheus style exposition format."""

    lines = []
    for key, value in metrics.items():
        metric_name = f"truth_api_{key}"
        lines.append(f"# TYPE {metric_name} gauge")
        lines.append(f"{metric_name} {value}")
    return "\n".join(lines) + "\n"


def ensure_directory(path: Path) -> None:
    """Ensure the directory for ``path`` exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
