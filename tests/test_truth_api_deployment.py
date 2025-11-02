from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tessrax_truth_api import deployment


def _write_tls_material(tmp_path: Path) -> tuple[Path, Path]:
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("dummy-cert", encoding="utf-8")
    key.write_text("dummy-key", encoding="utf-8")
    return cert, key


def test_load_deployment_settings_requires_tls(tmp_path: Path) -> None:
    config = {
        "deployment": {
            "host": "0.0.0.0",
            "port": 9443,
            "allowed_hosts": ["127.0.0.1"],
            "tls": {"cert_file": str(tmp_path / "missing.pem"), "key_file": str(tmp_path / "missing.key")},
        }
    }
    with pytest.raises(FileNotFoundError):
        deployment.load_deployment_settings(config)


def test_build_uvicorn_config_validates_tls(tmp_path: Path) -> None:
    cert, key = _write_tls_material(tmp_path)
    settings = deployment.DeploymentSettings(
        host="0.0.0.0",
        port=9443,
        cert_file=cert,
        key_file=key,
        allowed_hosts=["127.0.0.1"],
    )

    with patch.object(deployment.ssl.SSLContext, "load_cert_chain") as mocked_load:
        mocked_load.return_value = None
        config = deployment.build_uvicorn_config(settings)
    mocked_load.assert_called_once()
    assert config.host == "0.0.0.0"
    assert config.port == 9443
    assert config.ssl_certfile == str(cert)


def test_run_public_server_raises_on_failure(tmp_path: Path) -> None:
    cert, key = _write_tls_material(tmp_path)
    settings = deployment.DeploymentSettings(
        host="0.0.0.0",
        port=9443,
        cert_file=cert,
        key_file=key,
        allowed_hosts=["127.0.0.1"],
    )

    with patch.object(deployment, "build_uvicorn_config") as build_config:
        dummy_config = MagicMock()
        build_config.return_value = dummy_config
        server = MagicMock()
        server.run.return_value = False
        with patch.object(deployment.uvicorn, "Server", return_value=server):
            with pytest.raises(RuntimeError):
                deployment.run_public_server(settings)
