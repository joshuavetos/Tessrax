"""
Minimal CSV prototype (<500 lines)

Demonstrates the primitive: generate candidate + contrast outputs,
verify candidate by contradiction test, emit receipt.
"""

import hashlib
import random
import json

# --- Helpers -----------------------------------------------------

def sha256(x: str) -> str:
    return hashlib.sha256(x.encode()).hexdigest()

def generate_candidate(x: str) -> str:
    # placeholder: model output
    return x.upper()

def generate_contrast(x: str) -> str:
    # placeholder: simple perturbation
    return x.lower()

def verify(candidate: str, contrast: str) -> bool:
    # naive verification rule: candidate must differ meaningfully
    return candidate != contrast

def receipt(input_text: str, candidate: str, contrast: str, verified: bool):
    return {
        "input_hash": sha256(input_text),
        "candidate_hash": sha256(candidate),
        "contrast_hash": sha256(contrast),
        "verified": verified
    }

# --- Demo loop ---------------------------------------------------

if __name__ == "__main__":
    inputs = ["cat", "dog", "banana", "apple"]

    for text in inputs:
        cand = generate_candidate(text)
        cont = generate_contrast(text)
        ver  = verify(cand, cont)
        r = receipt(text, cand, cont, ver)
        print(json.dumps(r, indent=2))