"""
verify_pipeline.py
------------------
Minimal verification prototype (~100 lines)

Simulates:
1. Candidate generation from input
2. Contrastive alternative
3. Naive contradiction verification
4. Deterministic cryptographic receipt
"""

import hashlib
import json
from typing import List


# ------------------------------------------------------------
# Hashing Utilities
# ------------------------------------------------------------

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------
# Core Functions
# ------------------------------------------------------------

def generate_candidate(input_text: str) -> str:
    """Simulated model output (placeholder)."""
    return input_text.upper()

def generate_contrast(input_text: str) -> str:
    """Simulated contrastive output (perturbation)."""
    return input_text.lower()

def verify_contradiction(candidate: str, contrast: str) -> bool:
    """
    Naive contradiction rule:
    Candidate must differ significantly from contrast.
    """
    return candidate.strip().lower() != contrast.strip().lower()


# ------------------------------------------------------------
# Receipt Structure
# ------------------------------------------------------------

def make_receipt(input_text: str, candidate: str, contrast: str, verified: bool) -> dict:
    return {
        "input_hash": sha256(input_text),
        "candidate_hash": sha256(candidate),
        "contrast_hash": sha256(contrast),
        "verified": verified
    }


# ------------------------------------------------------------
# Demo Loop
# ------------------------------------------------------------

def run_demo(inputs: List[str], log_file: str = None):
    for text in inputs:
        candidate = generate_candidate(text)
        contrast = generate_contrast(text)
        verified = verify_contradiction(candidate, contrast)
        result = make_receipt(text, candidate, contrast, verified)
        print(json.dumps(result, indent=2))

        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")


# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------

if __name__ == "__main__":
    test_inputs = ["cat", "dog", "banana", "apple"]
    run_demo(test_inputs, log_file=None)  # Set log_file="receipts.jsonl" to persist