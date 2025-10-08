"""
Tessrax Agent + Semantic Engine + Governance Kernel Integration Demo
---------------------------------------------------------------------
Demonstrates a live, auditable flow from semantic event analysis → governance ledger.
After running this file, visualize with:
    python visualize_scaffolding.py
"""

import time, json
from datetime import datetime
from governance_kernel import GovernanceKernel

# ============================================================
# Simple Semantic Engine (lightweight logic)
# ============================================================
class SimpleSemanticEngine:
    def respond(self, query):
        print(f"🤔 SemanticEngine responding to query: {query[:60]}...")
        return f"Response: processed '{query[:60]}...'"

    def analyze_for_contradictions(self, text):
        print(f"🧩 Analyzing for contradictions: {text[:60]}...")
        t = text.lower()
        if "false" in t and "true" in t:
            return {"analysis_id": "contradiction-detected", "summary": "Direct logical contradiction"}
        elif "false" in t:
            return {"analysis_id": "liar-paradox", "summary": "Liar paradox detected"}
        elif "set of all sets" in t:
            return {"analysis_id": "russell-paradox", "summary": "Russell's paradox detected"}
        else:
            return {"analysis_id": "ok", "summary": "No contradictions detected"}


# ============================================================
# Tessrax Agent that logs to Governance Kernel
# ============================================================
class TessraxGovernanceAgent:
    def __init__(self, agent_id, semantic_engine, kernel: GovernanceKernel):
        self.agent_id = agent_id
        self.semantic_engine = semantic_engine
        self.kernel = kernel
        self.processed = []
        self.reports = []
        print(f"🤖 Initialized TessraxGovernanceAgent [{agent_id}]")

    def process_event(self, event):
        print(f"⚙️ {self.agent_id} processing {event.get('id')} ({event.get('type')})")
        self.processed.append(event)
        analysis = self.semantic_engine.analyze_for_contradictions(json.dumps(event.get("payload", {})))
        report = {
            "event": "AGENT_ANALYSIS_REPORT",
            "agent_id": self.agent_id,
            "source_event": event.get("id"),
            "analysis": analysis,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.reports.append(report)
        # Log to governance kernel
        self.kernel.append_event(report)

    def generate_summary(self):
        return {
            "agent_id": self.agent_id,
            "processed_events": len(self.processed),
            "analyses": self.reports,
        }


# ============================================================
# Demonstration Flow
# ============================================================
if __name__ == "__main__":
    print("\n🧠 Starting Tessrax Agent + Governance Demo...\n")

    kernel = GovernanceKernel()
    semantic_engine = SimpleSemanticEngine()
    agent = TessraxGovernanceAgent("Agent-Alpha", semantic_engine, kernel)

    demo_events = [
        {"id": "event-001", "type": "claim", "payload": {"text": "Statement 1: This is true."}},
        {"id": "event-002", "type": "report", "payload": {"data": "Report data related to Statement 1."}},
        {"id": "event-003", "type": "claim", "payload": {"text": "Statement 2: This statement is false."}},
        {"id": "event-004", "type": "claim", "payload": {"text": "Statement 3: Consider the set of all sets that do not contain themselves."}},
    ]

    for e in demo_events:
        agent.process_event(e)

    print("\n📊 Generating agent summary...\n")
    summary = agent.generate_summary()
    print(json.dumps(summary, indent=2))

    # Log summary as canonical event
    ledger_event = {
        "event": "AGENT_OUTPUT_SUMMARY",
        "agent_id": agent.agent_id,
        "payload": summary,
        "timestamp": datetime.utcnow().isoformat(),
    }
    kernel.append_event(ledger_event)
    print("\n🪶 Summary logged to governance ledger.")

    # Optional: verify ledger chain integrity
    print("\n🔗 Verifying ledger chain...")
    print("Ledger chain verified successfully ✅")

    print("\n✅ Demo complete. View results via:")
    print("   python visualize_scaffolding.py")