"""
housing_primitives.py
Tessrax v0.1 — Governance primitives for metabolizing housing contradictions.
"""

import math

def durability_yield(age: int, lifespan: int, maintenance_cost: float) -> float:
    """Return ROI percentage of building longevity."""
    remaining = max(lifespan - age, 1)
    yield_pct = (remaining / lifespan) * (1 - maintenance_cost)
    return round(yield_pct * 100, 2)

def insurance_inversion(payout_rate: float, failure_rate: float) -> float:
    """Premium reduction proportional to durability margin."""
    ratio = max(0, 1 - failure_rate / payout_rate)
    return round(ratio * 0.25, 3)  # Max 25% reduction

def transaction_cost_visibility(avg_price: float, refi_per_decade: float) -> float:
    """Reveal hidden churn cost as % of property value lost to transactions."""
    churn_cost = (refi_per_decade * 0.02) * avg_price
    return round(churn_cost, 2)

def material_provenance_value(lifespan: int, sustainability_score: float) -> float:
    """Assign value based on material longevity and sustainability."""
    base = lifespan * (1 + sustainability_score)
    return round(base / 10, 2)

if __name__ == "__main__":
    print("Durability Yield:", durability_yield(20, 80, 0.15))
    print("Insurance Inversion:", insurance_inversion(0.05, 0.02))
    print("Churn Cost:", transaction_cost_visibility(400000, 3))
    print("Material Provenance:", material_provenance_value(70, 0.8))