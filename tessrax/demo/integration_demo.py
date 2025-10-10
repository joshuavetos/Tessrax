"""
Tessrax Integration Demo
------------------------

Runs an end-to-end test of the contradiction engine,
governance kernel, ledger, and one domain module.
"""

from tessrax.core.contradiction_engine import detect_contradictions, score_stability, log_to_ledger
from tessrax.core.governance_kernel import route
from tessrax.domains.housing.housing_contradiction_detector import HousingDomain
from tessrax.core.ledger import GovernanceLedger
from tessrax.core.reconciliation import reconcile


def run_demo():
    print("\n🧭 Tessrax Integration Demo Starting...\n")
    claims = [
        {"agent": "A", "claim": "The door is open."},
        {"agent": "B", "claim": "The door is closed."},
    ]
    graph = detect_contradictions(claims)
    stability = score_stability(graph)
    event = route(graph, stability)
    log_to_ledger(graph, stability)
    print("✅ Core pipeline executed.")

    domain = HousingDomain()
    result = domain.detect_contradictions()
    print(f"\n🏘️ Housing contradictions found: {result['count']}")

    ledger = GovernanceLedger()
    print("\n📜 Ledger verification:", "✅ OK" if ledger.verify_chain() else "❌ FAIL")

    report = reconcile()
    print("\n📊 Reconciliation report:", report)
    print("\n🎉 Tessrax Integration Demo Complete.\n")


if __name__ == "__main__":
    run_demo()