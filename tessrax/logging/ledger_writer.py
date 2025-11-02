"""Ledger persistence helpers with optional S3 replication."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

try:  # pragma: no cover - moto is optional at runtime
    from moto import mock_s3  # type: ignore
except ImportError:  # pragma: no cover - fallback when moto absent
    try:
        from moto import mock_aws as mock_s3  # type: ignore
    except ImportError:  # pragma: no cover - fallback when moto absent
        mock_s3 = None

from config_loader import CloudLoggingConfig


@dataclass
class _S3Runtime:
    """Internal runtime state for mockable S3 writers."""

    client: Any
    bucket: str
    key_prefix: str
    mock_ctx: Any | None = None


class S3LedgerWriter:
    """Replicate ledger entries to an S3 bucket or moto-backed mock."""

    def __init__(
        self,
        config: CloudLoggingConfig,
        key_prefix: str = "ledger",
    ) -> None:
        if config.provider.lower() != "s3":
            raise ValueError("S3LedgerWriter requires provider to be 's3'")
        self._config = config
        self._runtime = self._initialise_runtime(config, key_prefix)

    @staticmethod
    def _initialise_runtime(
        config: CloudLoggingConfig, key_prefix: str
    ) -> _S3Runtime:
        mock_ctx = None
        if config.use_mock:
            global mock_s3  # type: ignore[global-statement]
            if mock_s3 is None:
                try:
                    from moto import mock_s3 as moto_mock_s3  # type: ignore
                except ImportError:
                    try:
                        from moto import mock_aws as moto_mock_s3  # type: ignore
                    except ImportError as exc:  # pragma: no cover - installation guard
                        raise RuntimeError(
                            "moto is required for mock S3 logging but is not installed"
                        ) from exc
                mock_s3 = moto_mock_s3
            mock_ctx = mock_s3()
            mock_ctx.start()
        session = boto3.session.Session()
        client = session.client(
            "s3",
            region_name=config.region,
            endpoint_url=config.endpoint_url,
        )
        runtime = _S3Runtime(client=client, bucket=config.bucket, key_prefix=key_prefix, mock_ctx=mock_ctx)
        S3LedgerWriter._ensure_bucket(runtime)
        return runtime

    @staticmethod
    def _ensure_bucket(runtime: _S3Runtime) -> None:
        try:
            runtime.client.head_bucket(Bucket=runtime.bucket)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in {"404", "NoSuchBucket"}:
                runtime.client.create_bucket(Bucket=runtime.bucket)
            else:  # pragma: no cover - defensive branch
                raise

    def close(self) -> None:
        if self._runtime.mock_ctx is not None:
            self._runtime.mock_ctx.stop()
            self._runtime.mock_ctx = None

    def append_line(self, payload: str) -> None:
        if not isinstance(payload, str):
            raise TypeError("Payload must be a JSON-encoded string")
        key = self._build_key()
        try:
            response = self._runtime.client.get_object(
                Bucket=self._runtime.bucket, Key=key
            )
            existing = response["Body"].read().decode("utf-8")
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code not in {"404", "NoSuchKey"}:
                raise
            existing = ""
        body = existing + (payload if payload.endswith("\n") else payload + "\n")
        try:
            self._runtime.client.put_object(
                Bucket=self._runtime.bucket,
                Key=key,
                Body=body.encode("utf-8"),
            )
        except (ClientError, BotoCoreError) as exc:
            raise RuntimeError(f"Failed to write ledger payload to S3: {exc}")

    def _build_key(self) -> str:
        now = datetime.now(timezone.utc)
        return f"{self._runtime.key_prefix}/{now:%Y/%m/%d}.jsonl"


class LedgerWriter:
    """Persist ledger entries locally and optionally replicate to S3."""

    def __init__(
        self,
        ledger_path: Path,
        cloud_writer: S3LedgerWriter | None = None,
    ) -> None:
        self.ledger_path = ledger_path
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._cloud_writer = cloud_writer

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise TypeError("Ledger record must be a dictionary")
        payload = json.dumps(record, sort_keys=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        if self._cloud_writer is not None:
            self._cloud_writer.append_line(payload)
        return record

    def close(self) -> None:
        if self._cloud_writer is not None:
            self._cloud_writer.close()


__all__ = ["LedgerWriter", "S3LedgerWriter"]
