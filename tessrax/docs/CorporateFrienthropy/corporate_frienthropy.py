"""
Corporate Frienthropy Analyzer
Tracks decay between a company's chartered promises and its actual output.

Requirements:
    pip install pandas matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------
# CONFIG
# ------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
CSV_FILE = BASE_DIR / "data" / "corporate_frienthropy.csv"
REPORT_FILE = Path(__file__).with_name("corporate_frienthropy_report.csv")
PLOT_FILE = Path(__file__).with_name("corporate_frienthropy_plot.png")

# ------------------------
# LOAD & CLEAN DATA
# ------------------------
if not CSV_FILE.exists():
    raise FileNotFoundError(f"{CSV_FILE} not found; run analytics/frienthropy_decay.py first.")

df = pd.read_csv(CSV_FILE, parse_dates=["timestamp"])

numeric_cols = ["trust_decay", "governance_score", "anomaly_rate"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[col].isnull().any():
        raise ValueError(f"Column {col} contained non-numeric data")

# ------------------------
# CALCULATE DERIVED METRICS
# ------------------------
df = df.sort_values("timestamp")
df["governance_slack"] = (1.0 - df["governance_score"]).clip(lower=0, upper=1)
df["stability_index"] = 1.0 - df["trust_decay"].clip(lower=0, upper=1)

# ------------------------
# EXPORT REPORT
# ------------------------
period_df = df[[
    "timestamp",
    "trust_decay",
    "governance_slack",
    "anomaly_rate",
    "stability_index",
]].copy()

# ------------------------
# EXPORT REPORT
# ------------------------
period_df.to_csv(REPORT_FILE, index=False)
print(f"✅ Exported Corporate Frienthropy report → {REPORT_FILE}\n")
print(period_df)

# ------------------------
# PLOT DECAY CURVE
# ------------------------
plt.figure(figsize=(8, 4))
plt.plot(period_df["timestamp"], period_df["trust_decay"], marker="o", linewidth=2, label="Trust Decay")
plt.plot(
    period_df["timestamp"],
    period_df["governance_slack"],
    marker="s",
    linewidth=1.8,
    linestyle="--",
    label="Governance Slack",
)
plt.fill_between(period_df["timestamp"], period_df["anomaly_rate"], color="#f97316", alpha=0.2, label="Anomaly Rate")
plt.title("Corporate Frienthropy Index Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Risk Proportion")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
plt.show()

print(f"✅ Saved decay curve plot → {PLOT_FILE}")
