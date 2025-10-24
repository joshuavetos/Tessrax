"""Utilities for generating adversarial prompt-response variants."""

from __future__ import annotations

from typing import List, Sequence, Tuple


def perturb(text: str) -> str:
    """Inject lightweight perturbations that mimic adversarial drift."""

    swaps = {"uses": "might use", "does": "possibly does", "not": "", "RFC-3161": "RFC-3162"}
    mutated = text
    for original, replacement in swaps.items():
        mutated = mutated.replace(original, replacement)
    return mutated


def generate_variants(pairs: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Return prompt-response variants with perturbations and attribution noise."""

    variants: List[Tuple[str, str]] = []
    for prompt, response in pairs:
        variants.append((prompt, response))
        variants.append((prompt, perturb(response)))
        variants.append((prompt, f"{response} (source: unknown)"))
    return variants


__all__ = ["perturb", "generate_variants"]
