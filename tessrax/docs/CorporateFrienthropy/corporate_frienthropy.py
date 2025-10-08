"""
Corporate Frienthropy Analyzer
Tracks decay between a company’s chartered promises and its actual output.

Requirements:
    pip install pandas matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------
# CONFIG
# ------------------------
CSV_FILE = "company_frienthropy.csv"
REPORT_FILE = "cfi_report.csv"
PLOT_FILE = "cfi_plot.png"

# ------------------------
# LOAD & CLEAN DATA
# ------------------------
if not Path(CSV_FILE).exists():
    raise FileNotFoundError(f"{CSV_FILE} not found in working directory.")

df = pd.read_csv(CSV_FILE)

numeric_cols = ["target_value", "actual_value", "weight"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ensure dates
df["period_start"] = pd.to_datetime(df["period_start"])
df["period_end"] = pd.to_datetime(df["period_end"])

# ------------------------
# CALCULATE CONTRADICTIONS
# ------------------------
# contradiction = 1 - (actual / target)
df["contradiction"] = 1 - (df["actual_value"] / df["target_value"])
df["contradiction"] = df["contradiction"].clip(lower=0)  # no negatives

# weighted contribution
df["weighted"] = df["contradiction"] * df["weight"]

# ------------------------
# AGGREGATE BY PERIOD
# ------------------------
df["period"] = df["period_end"].dt.to_period("Q").dt.to_timestamp("Q")  # group by quarter
period_df = (
    df.groupby("period")[["weighted"]]
    .sum()
    .rename(columns={"weighted": "CFI"})
    .reset_index()
)

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
plt.plot(period_df["period"], period_df["CFI"], marker="o", linewidth=2)
plt.title("Corporate Frienthropy Index Over Time")
plt.xlabel("Quarter End")
plt.ylabel("CFI (0 = aligned • 1 = max decay)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
plt.show()

print(f"✅ Saved decay curve plot → {PLOT_FILE}")