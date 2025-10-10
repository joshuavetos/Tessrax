"""
attention_primitives.py
Tessrax v0.1 — Governance primitives for metabolizing attention contradictions.
"""

def focus_yield(reclaimed_min: int, productivity_gain: float) -> float:
    """Return % productivity gained by reclaiming focus time."""
    return round(min(1.0, (reclaimed_min / 60) * productivity_gain) * 100, 2)

def dopamine_dividend(wellbeing_gain: float, ad_loss: float) -> float:
    """Economic value of healthier engagement (normalized)."""
    return round(max(0, wellbeing_gain - ad_loss) * 100, 2)

def feed_entropy_index(unique_sources: int, total_posts: int) -> float:
    """Measures diversity vs. overload balance."""
    return round(unique_sources / max(total_posts, 1), 3)

def ad_revenue_inversion(revenue: float, churn_reduction: float) -> float:
    """Calculates offset between reduced ads and lower churn."""
    return round(revenue * (1 - churn_reduction), 2)

if __name__ == "__main__":
    print("Focus Yield:", focus_yield(45, 0.8))
    print("Dopamine Dividend:", dopamine_dividend(0.25, 0.1))
    print("Feed Entropy:", feed_entropy_index(300, 1200))
    print("Ad Revenue Inversion:", ad_revenue_inversion(1_000_000, 0.15))