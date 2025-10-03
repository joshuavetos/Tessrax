# RFC-0: Computational Receipts + Contrastive Self-Verification (CSV)

## Primitive
Every inference step produces both a **candidate output** and a **contrast output**, plus a contract affirming or refuting the candidate relative to the contrast.

## Falsifiable Test
The system fails if it cannot consistently emit valid contradiction-based verification pairs that distinguish correct from erroneous outputs under controlled datasets.

## Tradeoff (Scars)
- Latency & throughput overhead (~2x compute)
- False negatives in subtle cases
- Adversarial exploitation risk
- Increased resource consumption
- Verification logic complexity

## Acceptance Criteria
- Verification accuracy > baseline sanity checks
- Overhead <10x for critical domains
- Clear provenance for each receipt

## Inevitability Arc
- **Trigger**: Regulatory / liability demands
- **Network effect**: Trust networks form around verified outputs
- **Workflow shift**: Enables verifiable self-debugging and safe autonomy