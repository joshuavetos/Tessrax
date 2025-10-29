"""Corporate Frienthropy decay analysis utilities.

This script rebuilds a decay curve using the canonical Tessrax governance
columns and persists both a tabular summary and a PNG chart. It also
supports deterministic synthetic data generation when the CSV input is
missing so operators can verify the plotting pipeline locally.
"""

from __future__ import annotations

import json
from calendar import monthrange
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "corporate_frienthropy.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
REPORT_PATH = OUTPUT_DIR / "corporate_frienthropy_report.json"
PLOT_PATH = OUTPUT_DIR / "corporate_frienthropy_decay.png"
TABLE_PATH = OUTPUT_DIR / "corporate_frienthropy_summary.csv"


@dataclass(frozen=True)
class DecayPoint:
    """Represents a single point along the frienthropy decay timeline."""

    timestamp: datetime
    trust_decay: float
    governance_score: float
    anomaly_rate: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "trust_decay": round(self.trust_decay, 4),
            "governance_score": round(self.governance_score, 4),
            "anomaly_rate": round(self.anomaly_rate, 4),
        }


def _ensure_dataset(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, parse_dates=["timestamp"])
    else:
        synthetic = list(_synthesise_points())
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([point.to_dict() for point in synthetic]).to_csv(path, index=False)
        df = pd.DataFrame([point.to_dict() for point in synthetic])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _add_months(base: datetime, months: int) -> datetime:
    year = base.year + (base.month - 1 + months) // 12
    month = (base.month - 1 + months) % 12 + 1
    day = min(base.day, monthrange(year, month)[1])
    return base.replace(year=year, month=month, day=day)


def _synthesise_points() -> Iterable[DecayPoint]:
    base = datetime(2025, 1, 15)
    for month, trust_decay, governance_score, anomaly_rate in [
        (0, 0.18, 0.74, 0.09),
        (1, 0.22, 0.71, 0.11),
        (2, 0.27, 0.69, 0.14),
        (3, 0.31, 0.66, 0.16),
        (4, 0.29, 0.68, 0.13),
        (5, 0.24, 0.72, 0.10),
        (6, 0.21, 0.74, 0.08),
        (7, 0.19, 0.76, 0.07),
        (8, 0.17, 0.78, 0.06),
        (9, 0.16, 0.79, 0.05),
        (10, 0.15, 0.80, 0.05),
        (11, 0.14, 0.81, 0.04),
    ]:
        yield DecayPoint(
            timestamp=_add_months(base, month),
            trust_decay=trust_decay,
            governance_score=governance_score,
            anomaly_rate=anomaly_rate,
        )


def _validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["timestamp", "trust_decay", "governance_score", "anomaly_rate"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    for column in required[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isnull().any():
            raise ValueError(f"Column {column} contains non-numeric values")
    return df.sort_values("timestamp")


def _persist_outputs(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_PATH, index=False)
    summary = {
        "data_points": len(df),
        "mean_decay": float(df["trust_decay"].mean()),
        "latest_governance_score": float(df["governance_score"].iloc[-1]),
        "max_anomaly_rate": float(df["anomaly_rate"].max()),
        "source_csv": str(DATA_PATH.relative_to(Path.cwd())),
    }
    REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _plot_decay(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 4.5))
    plt.plot(df["timestamp"], df["trust_decay"], marker="o", linewidth=2, label="Trust Decay")
    plt.plot(
        df["timestamp"],
        1.0 - df["governance_score"],
        marker="s",
        linewidth=1.8,
        linestyle="--",
        label="Governance Slack",
    )
    plt.fill_between(df["timestamp"], df["anomaly_rate"], color="#f97316", alpha=0.2, label="Anomaly Rate")
    plt.title("Corporate Frienthropy Decay Curve")
    plt.xlabel("Timestamp")
    plt.ylabel("Proportion of Risk")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


def main() -> None:
    dataset = _ensure_dataset(DATA_PATH)
    dataset = _validate_columns(dataset)
    _persist_outputs(dataset)
    _plot_decay(dataset)
    print(f"Report saved to {REPORT_PATH}")
    print(f"CSV export saved to {TABLE_PATH}")
    print(f"Decay curve saved to {PLOT_PATH}")


if __name__ == "__main__":
    main()
