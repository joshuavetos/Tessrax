# Demonstrate initializing and using the agent and semantic engine, and logging to a mock ledger

import time # Ensure time is imported for timestamp
import json # Ensure json is imported for payload serialization

# Ensure the necessary classes are available (defined in previous cells)
# SimpleSemanticEngine defined in J73PLwUWSWF1
# TessraxTestAgent defined in DMGvroJ2SYQE
# MockLedger defined in wBiQEg1ZRJm3 (or included locally if needed)

# Initialize mock dependencies
# Use the mock ledger from wBiQEg1ZRJm3 if available, otherwise create a new one
if 'MockLedger' in globals():
    print("Using MockLedger from global scope.")
    mock_ledger_instance = MockLedger("agent_demo.db")
else:
    # Basic in-memory mock if the main mock isn't available
    print("MockLedger not found globally, using basic fallback.")
    class BasicMockLedger:
        def __init__(self, *args, **kwargs):
            print("Using BasicMockLedger (fallback).")
            self._events = []
        def add_event(self, event):
            print(f"BasicMockLedger adding event: {event.get('type')}")
            # Add a timestamp if missing
            if 'timestamp' not in event:
                 event['timestamp'] = time.time()
            self._events.append(event)
        def get_all_events(self, verify=False):
             print("BasicMockLedger getting all events.")
             return self._events
        def verify_chain(self):
             print("BasicMockLedger verifying chain (always True).")
             return True

    mock_ledger_instance = BasicMockLedger("agent_demo.db")


# Initialize the semantic engine
if 'SimpleSemanticEngine' in globals():
    print("Using SimpleSemanticEngine from global scope.")
    semantic_engine_instance = SimpleSemanticEngine()
else:
    # Fallback if SimpleSemanticEngine is not defined
    print("SimpleSemanticEngine not found globally, using basic fallback.")
    class BasicSemanticEngine:
         def respond(self, query):
              print(f"BasicSemanticEngine acknowledging: {query[:50]}...")
              return f"Acknowledged query: {query[:50]}..."
         def analyze_for_contradictions(self, text):
              print(f"BasicSemanticEngine analyzing: {text[:50]}...")
              return {"analysis_id": "basic-mock-analysis", "summary": "Basic analysis performed."}

    semantic_engine_instance = BasicSemanticEngine()


# Initialize the test agent
if 'TessraxTestAgent' in globals() and isinstance(TessraxTestAgent, type):
    print("Using TessraxTestAgent from global scope.")
    test_agent = TessraxTestAgent("Agent-Alpha", semantic_engine_instance)
else:
    # Fallback if TessraxTestAgent is not defined or is not the expected type
    print("TessraxTestAgent not found globally, using basic fallback.")
    class BasicTestAgent:
         def __init__(self, agent_id, semantic_engine):
              print(f"Using BasicTestAgent (fallback) {agent_id}.")
              self.agent_id = agent_id
              self.semantic_engine = semantic_engine
              self._processed_events = []
              self._analysis_reports = []

         def process_event(self, event):
              print(f"BasicTestAgent {self.agent_id} processing event: {event.get('id', 'N/A')}")
              self._processed_events.append(event)
              if self.semantic_engine and hasattr(self.semantic_engine, 'analyze_for_contradictions'):
                   try:
                        analysis = self.semantic_engine.analyze_for_contradictions(json.dumps(event.get('payload', {})))
                        self._analysis_reports.append(analysis)
                   except Exception as e:
                        print(f"BasicTestAgent failed to analyze event: {e}")
              print("BasicTestAgent processed event.")

         def generate_output(self):
              output = f"BasicTestAgent {self.agent_id} processed {len(self._processed_events)} events."
              if self._analysis_reports:
                   output += f" Performed {len(self._analysis_reports)} analyses with results: {self._analysis_reports}" # Include reports in output for demo

              print("BasicTestAgent generated output.")
              return output

    # Need a dummy semantic engine if the real one wasn't found for the fallback agent
    if 'semantic_engine_instance' not in locals() or semantic_engine_instance is None:
         class DummySemanticEngineForFallback:
              def analyze_for_contradictions(self, text):
                   return {"analysis_id": "dummy-analysis", "summary": "No real analysis."}
         semantic_engine_instance = DummySemanticEngineForFallback()
         print("Warning: Using dummy semantic engine for BasicTestAgent fallback.")


    test_agent = BasicTestAgent("Agent-Alpha-Fallback", semantic_engine_instance)


# Simulate processing some events
print("\nSimulating agent processing events...")
test_agent.process_event({"id": "event-001", "type": "claim", "payload": {"text": "Statement 1: This is true."}})
test_agent.process_event({"id": "event-002", "type": "report", "payload": {"data": "Report data related to Statement 1."}})
# Add events that the SimpleSemanticEngine is designed to respond to/analyze
test_agent.process_event({"id": "event-003", "type": "claim", "payload": {"text": "Statement 2: This statement is false."}}) # Liar Paradox example
test_agent.process_event({"id": "event-004", "type": "claim", "payload": {"text": "Statement 3: Consider the set of all sets that do not contain themselves."}}) # Russell's Paradox example


# Generate and print the agent's output
print("\nGenerating agent output...")
agent_output = test_agent.generate_output()
print("\n--- Agent Generated Output ---")
print(agent_output)

# Now, add the agent's output as an event to the mock ledger
print("\nAdding agent output as an event to the mock ledger...")
ledger_event = {
    "type": "agent_output",
    "agent_id": test_agent.agent_id,
    # Include the full output in the payload for demonstration
    "payload": {"output_report": agent_output},
    "timestamp": time.time() # Add timestamp
}
try:
    mock_ledger_instance.add_event(ledger_event)
    print("Agent output event added to mock ledger.")
except Exception as e:
     print(f"Error adding event to mock ledger: {e}")


# Print the final state of the mock ledger
print("\n--- Final state of mock ledger ---")
try:
    all_ledger_events = mock_ledger_instance.get_all_events()
    # Pretty print the ledger events
    print(json.dumps(all_ledger_events, indent=2))
except Exception as e:
     print(f"Error getting events from mock ledger: {e}")
