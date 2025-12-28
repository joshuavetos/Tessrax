from pathlib import Path
from datetime import datetime, timezone

from tessrax.config_loader import CloudLoggingConfig
from tessrax.logging.ledger_writer import LedgerWriter, S3LedgerWriter


def test_s3_mock_replication(tmp_path: Path) -> None:
    cloud_config = CloudLoggingConfig(
        enabled=True,
        provider="s3",
        bucket="test-ledger",
        region="us-east-1",
        endpoint_url=None,
        use_mock=True,
    )
    cloud_writer = S3LedgerWriter(cloud_config)
    ledger_file = tmp_path / "ledger.jsonl"
    writer = LedgerWriter(ledger_file, cloud_writer)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stability_score": 0.87,
        "governance_lane": "deliberative",
        "claims": [{"agent": "test", "claim": "alpha"}],
    }
    writer.append(record)
    response = cloud_writer._runtime.client.get_object(  # type: ignore[attr-defined]
        Bucket=cloud_config.bucket,
        Key=cloud_writer._build_key(),
    )
    payload = response["Body"].read().decode("utf-8")
    assert "deliberative" in payload
    writer.close()
    assert ledger_file.exists()
    assert ledger_file.read_text(encoding="utf-8").strip() != ""


def test_writer_rejects_non_dict(tmp_path: Path) -> None:
    writer = LedgerWriter(tmp_path / "ledger.jsonl", None)
    try:
        writer.append([1, 2, 3])  # type: ignore[arg-type]
    except TypeError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("LedgerWriter should reject non-dict payloads")
