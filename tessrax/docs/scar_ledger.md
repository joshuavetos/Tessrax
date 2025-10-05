# Tessrax: The Scar Ledger

**Version:** 1.1  
**Status:** Draft Specification  
**Last Updated:** 2025-10-05  

---

## 1. Purpose

The **Scar Ledger** is the canonical archive of contradictions that have been detected and resolved within a Tessrax deployment.  
Each scar is both a historical receipt and a behavioral vaccine: proof that the system learned.

---

## 2. Structure

Each entry is a JSON object appended to the ledger with the following fields:

```json
{
  "id": "SCAR-2025-00041",
  "timestamp": "2025-10-05T12:30:44Z",
  "origin_event": "<hash of original contradiction>",
  "resolution": "quarantine | reform | rollback | policy_patch",
  "summary": "Description of contradiction and its metabolized outcome.",
  "author": "engine | human | hybrid",
  "signatures": ["<engine_sig>", "<reviewer_sig>"],
  "verdict_hash": "<sha256 of resolution payload>"
}