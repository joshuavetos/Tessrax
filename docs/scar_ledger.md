# Scar Ledger for CSV Primitive

## Scar #1: Latency / Throughput Bottleneck
- **Failure**: Dual outputs slow inference, unacceptable for real-time systems.
- **Mitigation ceiling**: Hardware acceleration; never zero cost.
- **Acceptance criterion**: Latency within domain thresholds.

## Scar #2: False Negatives
- **Failure**: Verification misses subtle errors → false confidence.
- **Mitigation ceiling**: Better training + rule design; no perfect detection.
- **Acceptance criterion**: False negatives tolerable at risk budget.

## Scar #3: Adversarial Exploitation
- **Failure**: Inputs trick CSV into passing harmful outputs.
- **Mitigation ceiling**: Adversarial training + external audits.
- **Acceptance criterion**: Residual success negligible + monitored.

## Scar #4: Resource Exhaustion
- **Failure**: Doubling compute cost limits deployment scale.
- **Acceptance criterion**: Overhead balanced by increased trust ROI.

## Scar #5: Verification Complexity
- **Failure**: Single atomic rule grows complex in practice.
- **Mitigation ceiling**: Formal specs, standard contracts.
- **Acceptance criterion**: Logic auditable + under error budget.