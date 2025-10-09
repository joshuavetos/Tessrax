# Resilience Ledger Template (v1.0)

**Schema**

| Date | CrisisType | Trigger | TripwireActivated | ResponseAction | Outcome | AuditRef | LessonsLearned |
|------|-------------|----------|-------------------|----------------|----------|-----------|----------------|

**Example Entry**

| 2025-06-15 | Financial Opacity | $2500 expense w/o dual sign-off | Dual-control tripwire | Payment frozen + audit launched | Funds protected | Audit#2025Q2 | Added stricter receipt policy |

**Purpose:** Continuous log of crises, tripwires, and corrective actions.  
Every event leaves a receipt; nothing erased, only metabolized into lessons.