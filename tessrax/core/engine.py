# Corrected content of tessrax.py for copying:

#!/usr/bin/env python3
"""
Tessrax Core Engine — Unified Contradiction + Graph Module
----------------------------------------------------------
Now includes a simple Quick-Start entry point so a new user can
run one end-to-end contradiction metabolism demo with a single command.

Usage:
    python /path/to/tessrax.py
"""

import sys
import os

# Add the directory containing the 'tessrax' and 'domains' folders to sys.path
# Assuming the project root is at /content/Tessrax-main/Tessrax-main/
project_root_for_path = "/content/Tessrax-main/Tessrax-main/"
if project_root_for_path not in sys.path:
    sys.path.insert(0, project_root_for_path)
    # print(f"Added {project_root_for_path} to sys.path for module discovery.") # Keep print optional
else:
    pass
    # print(f"{project_root_for_path} already in sys.path.") # Keep print optional


import hashlib, json
from collections import defaultdict
from typing import List, Dict, Any
import networkx as nx
import time # Import time module
# import needed modules from tessrax.core explicitly for clarity
# Assuming these imports are correct relative to the project root added to sys.path
from tessrax.core.contradiction_engine import detect_contradictions as run_contradiction_cycle # Corrected import based on file names
from tessrax.core.governance_kernel import route # Assuming this is the correct import path based on project structure

# Import from domains package
from domains.housing.housing_contradiction_detector import detect_contradictions as detect_housing_contradictions # Renamed to avoid conflict


# ----------------------------------------------------------------------
# Quick-Start
# ----------------------------------------------------------------------

def quickstart():
    """Run a minimal demonstration of Tessrax."""
    print("\n🧭  Tessrax Quick-Start Demo")
    demo_claims = [
        # These claims are simple and don't trigger the domains.housing detector,
        # but the import is still required.
        {"agent": "GPT", "claim": "A", "type": "normative"},
        {"agent": "Claude", "claim": "B", "type": "normative"},
        {"agent": "Gemini", "claim": "A", "type": "normative"},
    ]
    # Note: The original code called run_contradiction_cycle with demo_claims.
    # If detect_contradictions from domains was meant to be used here,
    # the logic would need to be adjusted. Assuming run_contradiction_cycle from contradiction_engine is the intended core function.
    # Also, the demo_claims are very simple and won't trigger the basic contradiction logic in contradiction_engine.
    # Let's use a slightly better example that might trigger the negation check.
    demo_claims_for_test = [
        {"agent": "Alice", "claim": "The door is open.", "type": "fact"},
        {"agent": "Bob", "claim": "The door is not open.", "type": "fact"},
        {"agent": "Charlie", "claim": "The cat is on the mat.", "type": "fact"},
        {"agent": "HousingAgent", "claim": "Property values are increasing.", "type": "fact"},
    ]


    # The quickstart function in the original tessrax.py called run_contradiction_cycle
    # which is the detect_contradictions function from contradiction_engine.py.
    # It also imported detect_contradictions from domains.housing but didn't seem to use it directly in quickstart.
    # Let's refine the quickstart to demonstrate calling the core contradiction engine
    # and potentially the domain-specific one if applicable, though the demo claims are simple.

    print("\nRunning core contradiction detection...")
    # Calling the detect_contradictions from contradiction_engine (aliased as run_contradiction_cycle)
    contradiction_graph_core = run_contradiction_cycle(demo_claims_for_test)
    stability_core = score_stability(contradiction_graph_core) # Need to import score_stability too
    print(f"Core stability score: {stability_core:.2f}")


    # If you intended to use the domain-specific detector, you would call it here.
    # For example:
    # housing_claims = [{"agent": "HousingAgent", "claim": "Property values are increasing.", "type": "fact"}]
    # housing_contradictions = detect_housing_contradictions(housing_claims)
    # print(f"\nHousing domain detector output: {housing_contradictions}")


    # Routing the core contradiction report...
    print("\nRouting core contradiction report...")
    # The route function in governance_kernel.py takes a graph and stability.
    # Need to pass the core graph and stability.
    event = route(contradiction_graph_core, stability_core)


    # event is likely an object, __dict__ might not be the intended way to serialize it.
    # If 'event' is an instance of a custom class, it needs to be serializable.
    # Assuming it's a simple object or has a __dict__ representation that works with json.
    try:
        # Check if event is a dataclass instance or has __dict__
        if hasattr(event, '__dataclass_fields__'): # Check if it's a dataclass
             event_dict = event.__dict__
        elif isinstance(event, dict): # Check if it's already a dict
             event_dict = event
        elif hasattr(event, '__dict__'): # Fallback to __dict__
             event_dict = event.__dict__
        else:
             event_dict = {"message": "Event object is not directly serializable to JSON.", "event_representation": str(event)} # Provide a string representation

        print(json.dumps(event_dict, indent=2))

    except TypeError:
        print("Could not serialize event object to JSON. Event object details:")
        print(event)


    # Log to a temporary ledger file for this example run
    # Need to import log_to_ledger and Path
    from tessrax.core.contradiction_engine import log_to_ledger # Assuming log_to_ledger is here
    from pathlib import Path
    log_to_ledger(contradiction_graph_core, stability_core, Path("data/governance_ledger.jsonl")) # Use core graph and stability


    print("\n✅  Demo complete.  Ledger updated at data/governance_ledger.jsonl\n")

# Need to import score_stability from contradiction_engine.py
from tessrax.core.contradiction_engine import score_stability


if __name__ == "__main__":
    # Detect environment variable QUICKSTART=0 to skip auto-demo (for CI)
    if os.getenv("QUICKSTART", "1") != "0":
        quickstart()
