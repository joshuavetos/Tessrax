"""Ledger persistence helpers with optional S3 replication."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

try:  # pragma: no cover - moto is optional at runtime
    from moto import mock_s3  # type: ignore
except ImportError:  # pragma: no cover - fallback when moto absent
    try:
        from moto import mock_aws as mock_s3  # type: ignore
    except ImportError:  # pragma: no cover - fallback when moto absent
        mock_s3 = None

# Internal project imports
try:
    from config_loader import CloudLoggingConfig
except ImportError:
    class CloudLoggingConfig:
        def __init__(self):
            self.s3_bucket = None
            self.local_path = "./ledger_storage"

@dataclass
class _InMemoryS3Client:
    """Lightweight in-memory stand-in for S3 when moto is unavailable."""
    def __init__(self) -> None:
        self._buckets: Dict[str, Dict[str, bytes]] = {}

    @staticmethod
    def _error(code: str, message: str, operation: str) -> ClientError:
        return ClientError({"Error": {"Code": code, "Message": message}}, operation)

    def head_bucket(self, Bucket: str) -> None:
        if Bucket not in self._buckets:
            raise self._error("404", "Bucket does not exist", "HeadBucket")

    def create_bucket(self, Bucket: str) -> None:
        self._buckets.setdefault(Bucket, {})

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        if Bucket not in self._buckets:
            raise self._error("404", "Bucket does not exist", "GetObject")
        bucket = self._buckets[Bucket]
        if Key not in bucket:
            raise self._error("NoSuchKey", "The specified key does not exist", "GetObject")
        return {"Body": BytesIO(bucket[Key])}

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> None:
        bucket = self._buckets.setdefault(Bucket, {})
        bucket[Key] = Body

class Ledger:
    """
    Persistence layer for Tessrax governance receipts.
    Supports local filesystem and optional S3 cloud replication.
    """
    def __init__(self, config: Optional[CloudLoggingConfig] = None):
        self.config = config or CloudLoggingConfig()
        self.logger = logging.getLogger("tessrax.ledger")
        self.local_dir = Path(getattr(self.config, 'local_path', './ledger'))
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 if configured
        self.s3_client = None
        if hasattr(self.config, 's3_bucket') and self.config.s3_bucket:
            try:
                self.s3_client = boto3.client('s3')
            except (BotoCoreError, ClientError) as e:
                self.logger.warning(f"Failed to init S3, falling back to local: {e}")

    def write_receipt(self, receipt: Dict[str, Any]) -> str:
        """
        Persists a reconciliation receipt to the ledger.
        Ensures dictionary data is serialized to JSON.
        """
        receipt_id = receipt.get("dispute_id", f"rcpt_{int(datetime.now().timestamp())}")
        filename = f"{receipt_id}.json"
        
        # Ensure timestamp is present
        if "resolved_at" not in receipt:
            receipt["resolved_at"] = datetime.now(timezone.utc).isoformat()

        content = json.dumps(receipt, indent=2)
        
        # 1. Local Write
        local_path = self.local_dir / filename
        with open(local_path, "w") as f:
            f.write(content)

        # 2. Optional Cloud Replication
        if self.s3_client and self.config.s3_bucket:
            try:
                self.s3_client.put_object(
                    Bucket=self.config.s3_bucket,
                    Key=f"receipts/{filename}",
                    Body=content.encode('utf-8')
                )
            except (BotoCoreError, ClientError) as e:
                self.logger.error(f"Cloud replication failed for {receipt_id}: {e}")

        return str(local_path)

    def verify_integrity(self, receipt_id: str) -> bool:
        """
        Placeholder for cryptographic verification.
        Checks if the receipt exists and is valid JSON.
        """
        local_path = self.local_dir / f"{receipt_id}.json"
        if not local_path.exists():
            return False
        
        try:
            with open(local_path, "r") as f:
                json.load(f)
            return True
        except json.JSONDecodeError:
            return False

if __name__ == "__main__":
    # Basic sanity test
    ledger = Ledger()
    test_receipt = {
        "dispute_id": "test_001",
        "final_value": 42.0,
        "status": "reconciled"
    }
    path = ledger.write_receipt(test_receipt)
    print(f"Receipt written to: {path}")

