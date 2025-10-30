# Tessrax Epistemic Metrics Specification

This document formalises the epistemic health metrics that power the Tessrax
ledger, provides traceability to automated schema validation, and records the
approved tolerance regime for audit reproducibility.

## Metric Formulas

All metrics operate on bounded numeric domains and clamp the output to the
``[0, 1]`` interval.  The canonical Python implementations live in
`tessrax/metrics/epistemic_health.py`.

### Integrity

``Integrity = 1 - (σ_t / σ_max)``

* ``σ_t`` – population standard deviation of the observed outcomes.
* ``σ_max`` – admissible maximum spread. Defaults to half of the observed range
  when not specified.

### Drift

``Drift = |x_t - mean(x_1 … x_{t-1})|``

* ``x_t`` – most recent value.
* ``mean`` – arithmetic mean of previous values.

### Severity

``Severity = mean(|expected_i - observed_i|)``

The metric evaluates the mean absolute error across matched pairs.

### Independence (Normalised Entropy)

``Independence = H(labels) / H_max`` where ``H`` is Shannon entropy.

* ``H`` – ``-Σ p_i log₂ p_i`` for each label proportion ``p_i``.
* ``H_max`` – ``log₂ n`` when ``n`` distinct labels are present, otherwise 1.0.

## Schema Specification

The `schemas/epistemic_metrics.schema.json` file captures the canonical shape of
metric payloads.  Every field is unit tested in `tests/test_metric_schema.py`
using JSON Schema Draft 2020-12 validation with deterministic test vectors.

## Provenance Workflow

The `verify-metric-provenance` GitHub Action establishes a baseline environment
snapshot via `tools/generate_env_snapshot.py`.  During the first run on a branch
without `out/env_snapshot.json` the action records the generated baseline as a
workflow artifact.  Subsequent runs compare the newly generated snapshot to the
baseline with these rules:

* **Exact match** is required for string, version, and hash values.
* **Numeric fields** tolerate a ±0.01 delta to account for floating-point jitter.
* **Regressions** beyond the tolerance thresholds trigger a `needs_reaudit`
  annotation instead of failing the pipeline to signal manual investigation.

### Manual Overrides

In rare cases, Gemini Governance may approve a manual override.  The authorised
engineer must:

1. Update `out/env_snapshot.json` with a signed commit describing the variance.
2. Reference the governance ticket in the commit message and ledger entry.
3. Attach a note to the action run documenting the override and auditor approval.

## Test Vector Linkage

The test vectors that back the schema live in `tests/data/epistemic_metrics/` and
are consumed by both the schema validation tests and the provenance workflow.
Each vector includes the expected Integrity, Drift, Severity, and Independence
values calculated by `tessrax/metrics/epistemic_health.py`.
