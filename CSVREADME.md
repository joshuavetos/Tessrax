# Contrastive Self-Verification (CSV)

## Overview
CSV (Contrastive Self-Verification) is a proposed **atomic AI primitive** that forces a model to generate a candidate output *and* a contrasting counter-output, then verify the candidate against the contrast.  
This embeds falsifiable, real-time self-assessment directly into inference.

## Why It Matters
- **Bottleneck**: Current AI lacks atomic self-verification, leading to uncontrolled error propagation.
- **Primitive**: Every inference step must emit `(candidate, contrast, verification)`.
- **Scars**: Adds latency and resource overhead, but accepts these costs for higher trust.
- **Inevitability**: Regulatory pressure + trust networks will make this the standard baseline for reliable AI.

## Repository Structure
- `README.md` — High-level overview
- `rfc/RFC-0.md` — Minimal spec + scar ledger
- `prototypes/csv_sandbox.py` — Minimal <500 line prototype
- `docs/scar_ledger.md` — Canonical list of failure modes
- `docs/inevitability.md` — Adoption arc + triggers