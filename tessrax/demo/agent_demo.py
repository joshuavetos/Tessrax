"""
Demonstration: Initialize and use Tessrax agent + semantic engine + mock ledger.
This file can be run directly to see live interaction, analysis, and logging.
"""

import time, json

# ---------------------------------------------------------------------
# Mock Ledger
# ---------------------------------------------------------------------
class MockLedger:
    def __init__(self, path="agent_demo.db"):
        self.path = path
        self._events = []
        print(f"Initialized MockLedger at {path}")

    def add_event(self, event):
        event.setdefault("timestamp", time.time())
        self._events.append(event)
        print(f"🪶 Logged event type: {event['type']}")

    def get_all_events(self, verify=False):
        print("📜 Returning all ledger events")
        return self._events

    def verify_chain(self):
        print("✅ Mock ledger chain verification: PASSED")
        return True


# ---------------------------------------------------------------------
# Simple Semantic Engine
# ---------------------------------------------------------------------
class SimpleSemanticEngine:
    def respond(self, query):
        print(f"🤔 SemanticEngine responding to query: {query[:50]}...")
        return f"Response: processed '{query[:50]}...'"

    def analyze_for_contradictions(self, text):
        print(f"🧩 Analyzing for contradictions: {text[:50]}...")
        if "false" in text.lower() and "true" in text.lower():
            return {"analysis_id": "contradiction-detected", "summary": "Direct logical contradiction"}
        elif "false" in text.lower():
            return {"analysis_id": "liar-paradox", "summary": "Liar paradox detected"}
        elif "set of all sets" in text.lower():
            return {"analysis_id": "russell-paradox", "summary": "Russell's paradox detected"}
        else:
            return {"analysis_id": "ok", "summary": "No contradictions detected"}


# ---------------------------------------------------------------------
# Test Agent
# ---------------------------------------------------------------------
class TessraxTestAgent:
    def __init__(self, agent_id, semantic_engine):
        self.agent_id = agent_id
        self.semantic_engine = semantic_engine
        self._processed_events = []
        self._analysis_reports = []
        print(f"🤖 Initialized TessraxTestAgent {agent_id}")

    def process_event(self, event):
        print(f"⚙️ {self.agent_id} processing event {event.get('id')}")
        self._processed_events.append(event)
        analysis = self.semantic_engine.analyze_for_contradictions(json.dumps(event.get("payload", {})))
        self._analysis_reports.append(analysis)

    def generate_output(self):
        output = {
            "agent_id": self.agent_id,
            "events_processed": len(self._processed_events),
            "analyses": self._analysis_reports,
        }
        print(f"🧾 {self.agent_id} generated output for {len(self._processed_events)} events.")
        return json.dumps(output, indent=2)


# ---------------------------------------------------------------------
# Demonstration Run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ledger = MockLedger("agent_demo.db")
    semantic_engine = SimpleSemanticEngine()
    agent = TessraxTestAgent("Agent-Alpha", semantic_engine)

    # Simulated event stream
    demo_events = [
        {"id": "event-001", "type": "claim", "payload": {"text": "Statement 1: This is true."}},
        {"id": "event-002", "type": "report", "payload": {"data": "Report data related to Statement 1."}},
        {"id": "event-003", "type": "claim", "payload": {"text": "Statement 2: This statement is false."}},
        {"id": "event-004", "type": "claim", "payload": {"text": "Statement 3: Consider the set of all sets that do not contain themselves."}},
    ]

    print("\n🧠 Processing events...\n")
    for e in demo_events:
        agent.process_event(e)

    print("\n📊 Generating agent output...\n")
    output = agent.generate_output()
    print(output)

    ledger_event = {
        "type": "agent_output",
        "agent_id": agent.agent_id,
        "payload": {"output_report": output},
        "timestamp": time.time(),
    }

    print("\n📥 Logging agent output to ledger...\n")
    ledger.add_event(ledger_event)

    print("\n📚 Final state of ledger:\n")
    print(json.dumps(ledger.get_all_events(), indent=2))