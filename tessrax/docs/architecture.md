Tessrax Architecture

This document describes the evolved architecture of Tessrax — a modular contradiction-metabolism framework that detects, records, and governs system contradictions across domains.

⸻

1. System Overview

Tessrax transforms contradictions into first-class data objects.
It detects logical, semantic, temporal, and normative conflicts between agents or systems, routes them through governance logic, and records the process in a tamper-evident ledger.

Core subsystems:
   •   Core Engine (tessrax/core) — foundational logic for contradiction detection, governance, and reconciliation.
   •   Domains (tessrax/domains) — plug-in detectors for specific contexts (e.g. housing, AI memory).
   •   Data (data/) — append-only ledgers and reconciliation artifacts.
   •   Docs & Demos (docs/, demo/) — documentation and runnable examples.

⸻

2. Core Primitives

2.1 Contradiction Engine
   •   Purpose: Detects and classifies contradictions between agent claims.
   •   Types: Logical, Temporal, Semantic, Normative.
   •   Outputs: A contradiction graph (networkx.Graph) with edges representing conflicts.
   •   Key Functions:
      •   detect_contradictions(agent_claims)
      •   score_stability(graph)
      •   log_to_ledger(graph, stability, path)

⸻

2.2 Governance Kernel
   •   Purpose: Routes contradiction data into the correct governance lane.
   •   Mechanism: Stability index → Governance lane.
   •   Lanes:
      •   Autonomic — high stability; automatic acceptance.
      •   Deliberative — medium disagreement; requires review.
      •   Constitutional — structural contradiction; requires charter update.
      •   Behavioral Audit — semantic or definitional manipulation detected.
   •   Key Functions:
      •   route(graph, stability_index)
      •   classify_lane(stability)

⸻

2.3 Ledger
   •   Purpose: Append-only, verifiable record of all governance events.
   •   Implementation: GovernanceLedger class with chained SHA-256 hashes.
   •   Features:
      •   Hash-linked entries (block-chain style integrity).
      •   Merkle verification via verify_chain().
      •   JSONL storage for lightweight auditing.
   •   Role: Provides auditability and temporal provenance for every governance event.

⸻

2.4 Interfaces
   •   Purpose: Define the required structure for any domain plug-in.
   •   Interface: DomainInterface with detect_contradictions() and optional serialization helpers.
   •   Role: Guarantees consistent behavior across independently developed domains.

⸻

2.5 Reconciliation
   •   Purpose: Periodically analyzes the ledger to detect resolved or recurring contradictions.
   •   Function: reconcile(ledger_path) summarizes unresolved contradictions and produces reconciliation statistics.
   •   Role: Enables adaptive governance evolution by measuring contradiction decay over time.

⸻

2.6 Domains
   •   Purpose: Provide specialized contradiction logic for different knowledge areas.
   •   Examples:
      •   housing — economic and policy contradictions.
      •   ai_memory — inconsistency across knowledge embeddings.
      •   attention — cognitive focus conflicts.
   •   Implementation: Each domain subclasses DomainInterface and registers itself automatically via load_domains().

⸻

3. Governance Pipeline (Runtime Flow)

+-------------+        +----------------+        +---------------+
| Agent Claims| -----> | Contradiction  | -----> | Governance    |
|  (Inputs)   |        |   Engine       |        |   Kernel      |
+-------------+        +----------------+        +---------------+
       |                         |                       |
       |                         v                       v
       |                   [Contradiction Graph]   [Governance Event]
       |                         |                       |
       v                         v                       v
+----------------+        +----------------+        +----------------+
| Ledger (JSONL) | -----> | Reconciliation | -----> |  Reports / API |
+----------------+        +----------------+        +----------------+


⸻

4. Extensibility Mechanism

Domain Discovery

load_domains() dynamically scans the domains/ directory and imports any detector implementing DomainInterface.

Configuration

Future configuration via charter.json will define:
   •   Governance thresholds (Autonomic, Deliberative, etc.)
   •   Active domains
   •   Ledger storage paths

Example Invocation

python core/engine.py                # Core demo
python core/engine.py --domain all   # Run all available domains
python core/engine.py --verify-ledger


⸻

5. Data Integrity Model
   •   Hash Chain: Every ledger entry includes prev_hash and hash.
   •   Tamper Detection: verify_chain() recomputes hashes to ensure integrity.
   •   Reconciliation: Summarizes contradiction persistence and decay metrics.
   •   Receipts: (future feature) Each ledger entry can be wrapped in a signed receipt for external anchoring.

⸻

6. Design Principles
   •   Auditability: Every computation and governance event is traceable.
   •   Composability: Domains and governance lanes are modular.
   •   Extensibility: New contradiction types and governance rules can be added dynamically.
   •   Transparency: Contradictions are surfaced, not hidden.
   •   Autonomy + Oversight: Machine decisions are routable through human-in-the-loop lanes.

⸻

7. Limitations & Future Work
   •   Reconciliation logic is minimal; lacks weighted persistence analysis.
   •   Governance kernel currently uses static thresholds instead of policy files.
   •   Only the housing domain is fully implemented; others remain stubs.
   •   Lacks full API layer for external integrations.
   •   Future milestones:
      •   Domain generalization (AI memory, attention, governance, climate)
      •   Visual ledger explorer (contradiction graph visualization)
      •   Policy charter import/export
      •   Optional distributed ledger anchoring

⸻

Last updated: 2025-10-10

