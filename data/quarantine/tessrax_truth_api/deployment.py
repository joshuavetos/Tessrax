"""Public deployment helpers for the Tessrax Truth API."""

from __future__ import annotations

import os
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn

from tessrax_truth_api.utils import load_config


@dataclass
class DeploymentSettings:
    """Runtime settings used for TLS verified public deployment."""

    host: str
    port: int
    cert_file: Path
    key_file: Path
    allowed_hosts: list[str]

    def validate(self) -> None:
        if not self.host:
            raise ValueError("Host must be provided for deployment")
        if not isinstance(self.port, int) or not (1 <= self.port <= 65535):
            raise ValueError("Port must be an integer between 1 and 65535")
        if not self.cert_file or not self.cert_file.exists():
            raise FileNotFoundError(f"TLS certificate not found: {self.cert_file}")
        if not self.key_file or not self.key_file.exists():
            raise FileNotFoundError(f"TLS key not found: {self.key_file}")
        if not self.allowed_hosts:
            raise ValueError("allowed_hosts must contain at least one entry")
        for host in self.allowed_hosts:
            if not host:
                raise ValueError("allowed_hosts entries must be non-empty strings")


def load_deployment_settings(config: dict[str, Any] | None = None) -> DeploymentSettings:
    """Load deployment settings from configuration or supplied mapping."""

    config = config or load_config()
    deployment = config.get("deployment", {})
    tls_config = deployment.get("tls", {})
    cert_value = str(tls_config.get("cert_file", "")).strip() or os.environ.get("TRUTH_API_TLS_CERT", "")
    key_value = str(tls_config.get("key_file", "")).strip() or os.environ.get("TRUTH_API_TLS_KEY", "")
    if not cert_value:
        raise ValueError("TLS certificate path is not configured")
    if not key_value:
        raise ValueError("TLS key path is not configured")
    cert_path = Path(cert_value).expanduser().resolve(strict=False)
    key_path = Path(key_value).expanduser().resolve(strict=False)
    settings = DeploymentSettings(
        host=str(deployment.get("host", "0.0.0.0")),
        port=int(deployment.get("port", 8443)),
        cert_file=cert_path,
        key_file=key_path,
        allowed_hosts=list(deployment.get("allowed_hosts", ["*"])),
    )
    settings.validate()
    return settings


def ensure_tls_pair(cert_file: Path, key_file: Path) -> None:
    """Verify the TLS certificate/key pair loads successfully."""

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    try:
        context.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))
    except ssl.SSLError as exc:  # TESST
        raise RuntimeError(f"Invalid TLS material: {exc}") from exc


def build_uvicorn_config(settings: DeploymentSettings) -> uvicorn.Config:
    """Create a uvicorn configuration using the validated settings."""

    ensure_tls_pair(settings.cert_file, settings.key_file)
    return uvicorn.Config(
        app="tessrax_truth_api.main:app",
        host=settings.host,
        port=settings.port,
        ssl_certfile=str(settings.cert_file),
        ssl_keyfile=str(settings.key_file),
        proxy_headers=True,
        forwarded_allow_ips=",".join(settings.allowed_hosts),
        log_level="info",
    )


def run_public_server(settings: DeploymentSettings | None = None) -> None:
    """Run the Truth API on a TLS verified public endpoint."""

    deployment_settings = settings or load_deployment_settings()
    config = build_uvicorn_config(deployment_settings)
    server = uvicorn.Server(config)
    if not server.run():  # pragma: no cover - uvicorn handles loop
        raise RuntimeError("Truth API server terminated unexpectedly")


__all__ = [
    "DeploymentSettings",
    "build_uvicorn_config",
    "ensure_tls_pair",
    "load_deployment_settings",
    "run_public_server",
]
