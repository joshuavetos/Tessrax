from pathlib import Path

import pytest

from scripts.etl_public_health import InfluenzaRecord, load_dataset, to_claims


def test_load_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text(
        "state,week_start,ili_percentage,lab_confirmed_flu_cases\n"
        "Oregon,2023-11-05,5.5,321\n",
        encoding="utf-8",
    )
    records = load_dataset(dataset)
    assert len(records) == 1
    assert records[0] == InfluenzaRecord("Oregon", "2023-11-05", 5.5, 321)


def test_to_claims_generates_messages() -> None:
    records = [InfluenzaRecord("Oregon", "2023-11-05", 5.5, 321)]
    claims = to_claims(records)
    assert len(claims) == 1
    claim = claims[0]
    assert claim["agent"] == "etl.public_health"
    assert "Oregon" in claim["claim"]
    assert claim["context"]["lab_confirmed_flu_cases"] == 321


def test_load_dataset_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.csv")
