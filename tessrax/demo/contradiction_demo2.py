"""
Tessrax Prototype: Corporate Pledge vs Outcome Auditor
------------------------------------------------------

A minimal contradiction-auditing pipeline.

Input:  Natural-language statements (e.g., pledges and reports)
Output: Structured claims, detected contradictions, cryptographic receipt, and a visual map.

Requires: pip install graphviz
"""

import re
import json
import hashlib
import datetime
from graphviz import Digraph

# ===============================================================
# 1. Claim Extraction
# ===============================================================

def extract_claims(texts):
    """
    Convert raw text statements into structured claim objects.
    Extremely naive — designed only for demonstration.
    """
    claims = []

    for i, text in enumerate(texts, start=1):
        subj = "Acme Corp" if "Acme" in text else "Unknown"

        # Predicate classification
        if any(k in text.lower() for k in ["pledge", "target", "goal"]):
            pred = "emissions_reduction_target"
        elif any(k in text.lower() for k in ["report", "actual", "achieved"]):
            pred = "emissions_reduction_actual"
        else:
            pred = "statement"

        # Extract numeric value (first % number)
        num_match = re.search(r"(\d+(?:\.\d+)?)\s*%?", text)
        value = float(num_match.group(1)) if num_match else None

        # Extract year or use today
        date_match = re.search(r"(20\d{2})", text)
        year = date_match.group(1) if date_match else str(datetime.date.today().year)
        date = f"{year}-01-01"

        claim = {
            "id": f"c{i}",
            "subject": subj,
            "predicate": pred,
            "value": value,
            "date": date,
            "text": text.strip()
        }
        claims.append(claim)

    return claims


# ===============================================================
# 2. Contradiction Detection
# ===============================================================

def detect_contradiction(claims, tolerance=5.0):
    """
    Detects contradiction between target and actual emission reduction values.
    Returns a structured contradiction record if the gap exceeds tolerance.
    """
    target = next((c for c in claims if "target" in c["predicate"]), None)
    actual = next((c for c in claims if "actual" in c["predicate"]), None)
    if not (target and actual):
        return None

    if target["value"] is None or actual["value"] is None:
        return None

    diff = abs(target["value"] - actual["value"])
    if diff <= tolerance:
        return None

    contradiction = {
        "description": f"Outcome diverges from pledge ({target['value']}% vs {actual['value']}%)",
        "claims": [target["id"], actual["id"]],
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

    bundle = json.dumps(contradiction, sort_keys=True).encode()
    contradiction["bundle_hash"] = "sha256:" + hashlib.sha256(bundle).hexdigest()
    return contradiction


# ===============================================================
# 3. Ledger Receipt (Mock)
# ===============================================================

def record_receipt(contradiction, filename="ledger_receipts.jsonl"):
    """
    Appends a contradiction receipt to a local ledger file.
    """
    if not contradiction:
        return None

    receipt = {
        "receipt_id": "rcpt_" + contradiction["bundle_hash"][-8:],
        "ledger": "Tessrax",
        **contradiction
    }

    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(receipt) + "\n")

    return receipt


# ===============================================================
# 4. Visualization
# ===============================================================

def draw_graph(claims, contradiction, output="contradiction_map"):
    """
    Creates a simple Graphviz diagram of the claims and contradiction link.
    """
    g = Digraph("ContradictionMap", format="png")
    g.attr("node", shape="box", style="filled", color="lightgrey")

    for c in claims:
        g.node(c["id"], f"{c['date']}: {c['text']}")

    if contradiction:
        g.node("X", f"❌ {contradiction['description']}", color="red", shape="ellipse")
        for cid in contradiction["claims"]:
            g.edge(cid, "X")

    g.render(output, cleanup=True)
    print(f"[✓] Graph rendered → {output}.png")


# ===============================================================
# 5. Demo Runner
# ===============================================================

if __name__ == "__main__":
    statements = [
        "In 2020, Acme Corp pledged to cut CO₂ emissions 50% by 2030.",
        "In 2024, Acme Corp reported CO₂ emissions only down 5%."
    ]

    # Step 1: Extract claims
    claims = extract_claims(statements)
    print("\nExtracted Claims:")
    print(json.dumps(claims, indent=2))

    # Step 2: Detect contradictions
    contradiction = detect_contradiction(claims, tolerance=5.0)
    if contradiction:
        print("\nDetected Contradiction:")
        print(json.dumps(contradiction, indent=2))
    else:
        print("\nNo contradictions detected.")

    # Step 3: Record ledger receipt
    receipt = record_receipt(contradiction)
    if receipt:
        print("\nLedger Receipt:")
        print(json.dumps(receipt, indent=2))
    else:
        print("\nNo receipt recorded.")

    # Step 4: Visualize
    draw_graph(claims, contradiction)