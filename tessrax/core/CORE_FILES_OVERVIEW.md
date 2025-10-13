# Tessrax Core Module Overview

This document captures a high-level audit of the primary runtime modules located in `tessrax/core`. It focuses on the entry points that orchestrate contradiction detection, routing, and ledger persistence.

## `engine_core.py`
- Provides a lightweight orchestration helper (`analyze`) that stitches together contradiction detection and governance routing for a list of agent claims.
- Builds an undirected NetworkX graph where each agent is a node and edges record whether two claims disagree, then scores graph stability as `1 - contradictions/edges`.
- Routes the graph to a governance "lane" (`autonomic`, `deliberative`, `constitutional`, `behavioral_audit`) based on stability thresholds before appending the result to `data/ledger.jsonl` with a timestamp, stability index, lane, and participating agents.
- Ensures the ledger directory exists prior to appending, but relies on newline-delimited JSON without hash chaining, so downstream integrity checking must happen elsewhere.

## `engine.py`
- Acts as the CLI entry point with argument parsing, demo execution, optional domain modules, and REPL/ledger inspection utilities.
- Injects a Colab-oriented project path (`/content/Tessrax-main/Tessrax-main/`) into `sys.path`, which is convenient for notebooks but may require overriding when running from a different checkout.
- Gracefully imports optional domain-specific detectors (housing, AI memory, attention, governance, climate) and tracks availability flags so commands can degrade when modules are missing.
- Bundles helper utilities for pretty-printing JSON, loading extra configuration, and checking ledger integrity through hash chaining.

## `contradiction_engine.py`
- Defines structured claim handling via a `Claim` dataclass, providing type enums, timestamps, and hashed IDs to make objects hashable for NetworkX graphs.
- Implements a dictionary-to-`Claim` conversion pipeline that currently performs minimal validation; missing `agent` or `claim` fields are skipped with a warning, while unknown claim types default to `UNKNOWN`.
- Current contradiction heuristics focus on simplistic negation detection between normative claims; fact and prediction comparisons are stubbed out, indicating room for richer semantic analysis.
- Offers `score_stability` (edge-density inverse) and `log_to_ledger`, which appends contradiction events to `data/governance_ledger.jsonl` without hash chaining. The latter writes timestamped summaries and can raise if the file is inaccessible.
- Ships a second section titled "Tessrax Metabolism Core — Contradiction Tensor" that introduces vector-based contradiction metrics (`contradiction_tensor`, `step`, etc.) for advanced simulations, including normalization, volatility capping, and temporal dynamics utilities.

## `governance_kernel.py`
- Declares governance lanes via an enum and encapsulates ledger entries in a `GovernanceEvent` dataclass.
- Integrates optional Redis-based locks (selected via the `REDIS_URL` environment variable) with a FileLock fallback to coordinate writes in concurrent environments.
- Classifies governance lanes using edge metadata and stability thresholds, producing human-readable summaries for audit trails.
- Persists events to `data/governance_ledger.jsonl` with deterministic SHA-256 hashes and previous-hash chaining, enabling append-only integrity verification when `route` is invoked.

## `ledger_core.py`
- Supplies standalone helpers to hash entries, append records with chain continuity, and verify the ledger hash chain stored at `data/ledger.jsonl`.
- Uses UTC ISO timestamps and ensures the target directory exists before writing, mirroring the approach taken in `engine_core`.

## `ledger.py`
- Wraps ledger handling in a `GovernanceLedger` class that maintains hash chaining for governance events, including methods for appending, reading all entries, and verifying the chain.
- Each entry stores the raw event payload under `event`, with hashes computed over the timestamp, previous hash, and event body for reproducible verification.

## Key Observations & Opportunities
- Multiple modules write to ledgers with slightly different schemas (`engine_core` vs. `governance_kernel` vs. `GovernanceLedger`), so downstream consumers should normalize event formats or rely on a single ledger implementation to avoid drift.
- Contradiction detection is intentionally conservative; expanding beyond negation checks (e.g., semantic similarity, numerical comparisons) would materially improve signal quality.
- The hard-coded Colab path in `engine.py` may warrant parameterization for portability, especially when the package is installed elsewhere.
- Where hash chaining is desirable, prefer the `governance_kernel` or `GovernanceLedger` pathways; the basic ledger appenders do not embed integrity metadata.
