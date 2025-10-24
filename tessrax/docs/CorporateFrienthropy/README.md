# Corporate Frienthropy Package

> Placeholder data for runtime stability. Replace with production ledger or dataset.

Frienthropy: borrowed from friendship entropy — measures the decay between what’s promised and what’s delivered.

This package tracks a company’s **Charter → Output** drift by scoring each promise on how far the actual results fall short of the target.

---

## Files
- **company_frienthropy.csv** – your input data (see template for headers).
- **corporate_frienthropy.py** – analytics script (computes scores, exports report, plots curve).
- **cfi_report.csv** – auto-generated summary of the Corporate Frienthropy Index (CFI) per period.
- **cfi_plot.png** – decay-curve visualization.

---

## Quick Start
1. `pip install pandas matplotlib`
2. Replace sample rows in `company_frienthropy.csv` with real company promises, metrics, and periods.
3. Run:
```bash
python corporate_frienthropy.py
4.	View the cfi_report.csv and the cfi_plot.png curve.

⸻

Metric Logic

For each promise i:
contradiction_i = 1 - (actual_value / target_value)
weighted_i      = contradiction_i * weight_i
Aggregate all weighted_i within a period to get the Corporate Frienthropy Index:
CFI_period = Σ weighted_i
   •   0 → words and deeds fully aligned
   •   1 → total failure to meet all promises

Plotting CFI over time shows the decay curve of mission integrity.