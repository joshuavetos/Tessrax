"""
Tessrax Colab Full Runtime Bundle
Merged from Colab_code_1.txt and Colab_code_2.txt
Contains 12 independent subsystems:
  1. PredictiveRuntime
  2. IntegrityMonitor
  3. ZKAudit
  4. ClosureLedger
  5. EntropyForecaster
  6. LedgerIntegrityEngine
  7. GovernanceKernelRuntime
  8. EntropySignalAnalyzer
  9. CCGNPrototype
  10. ForecastAgent
  11. ZKProofEngine
  12. CCGNVisualizer
"""

import os, json, time, math, random, hashlib, hmac, logging, datetime
import networkx as nx

try:
    import numpy as np
except ImportError:
    np = None

# ---------------------------------------------------------------------
# 1. PredictiveRuntime
# ---------------------------------------------------------------------
class PredictiveRuntime:
    def __init__(self):
        self.history = []

    def run_cycle(self, metric_value: float):
        prediction = self._predict(metric_value)
        self.history.append({"input": metric_value, "prediction": prediction})
        return prediction

    def _predict(self, value):
        # Simple sigmoid normalization to [0,1]
        return 1 / (1 + math.exp(-value))

# ---------------------------------------------------------------------
# 2. IntegrityMonitor
# ---------------------------------------------------------------------
class IntegrityMonitor:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def compute_hash(self, filepath):
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()

    def verify_directory(self):
        report = {}
        for root, _, files in os.walk(self.base_dir):
            for f in files:
                path = os.path.join(root, f)
                report[path] = self.compute_hash(path)
        return report

# ---------------------------------------------------------------------
# 3. ZKAudit
# ---------------------------------------------------------------------
class ZKAudit:
    def __init__(self, secret_key: bytes):
        self.secret_key = secret_key

    def commit(self, message: str):
        digest = hmac.new(self.secret_key, message.encode(), hashlib.sha256).hexdigest()
        return {"message_hash": digest, "timestamp": datetime.datetime.utcnow().isoformat()}

# ---------------------------------------------------------------------
# 4. ClosureLedger
# ---------------------------------------------------------------------
class ClosureLedger:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_event(self, event_id, causes):
        self.graph.add_node(event_id, status="open", resolution=None)
        for c in causes:
            self.graph.add_edge(c, event_id)

    def close_event(self, event_id, resolution):
        if not self.graph.has_node(event_id):
            raise ValueError("Event not found")
        self.graph.nodes[event_id]["status"] = "closed"
        self.graph.nodes[event_id]["resolution"] = resolution

    def export_json(self):
        return json.dumps(nx.node_link_data(self.graph), indent=2)

# ---------------------------------------------------------------------
# 5. EntropyForecaster
# ---------------------------------------------------------------------
class EntropyForecaster:
    def __init__(self):
        self.values = []

    def update(self, val):
        self.values.append(val)
        return self.predict_next()

    def predict_next(self):
        if len(self.values) < 2:
            return self.values[-1] if self.values else 0
        diffs = [self.values[i+1]-self.values[i] for i in range(len(self.values)-1)]
        avg_delta = sum(diffs)/len(diffs)
        return self.values[-1] + avg_delta

# ---------------------------------------------------------------------
# 6. LedgerIntegrityEngine
# ---------------------------------------------------------------------
class LedgerIntegrityEngine:
    def __init__(self):
        self.ledger = []

    def append_entry(self, entry: dict):
        entry_json = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_json.encode()).hexdigest()
        self.ledger.append({"entry": entry, "hash": entry_hash})
        return entry_hash

    def verify_ledger(self):
        return all(
            rec["hash"] == hashlib.sha256(json.dumps(rec["entry"], sort_keys=True).encode()).hexdigest()
            for rec in self.ledger
        )

# ---------------------------------------------------------------------
# 7. GovernanceKernelRuntime
# ---------------------------------------------------------------------
class GovernanceKernelRuntime:
    def __init__(self):
        self.events = []
        self.logger = logging.getLogger("GovernanceKernel")
        logging.basicConfig(level=logging.INFO)

    def detect_contradiction(self, signal_strength):
        contradiction = signal_strength > 0.7
        self.logger.info(f"Signal={signal_strength:.2f} -> Contradiction={contradiction}")
        self.events.append({"signal": signal_strength, "contradiction": contradiction})
        return contradiction

# ---------------------------------------------------------------------
# 8. EntropySignalAnalyzer
# ---------------------------------------------------------------------
class EntropySignalAnalyzer:
    def __init__(self, window=5):
        self.window = window
        self.data = []

    def push(self, val):
        self.data.append(val)
        if len(self.data) > self.window:
            self.data.pop(0)
        return self.compute_entropy()

    def compute_entropy(self):
        if not self.data:
            return 0
        # Handle potential division by zero or negative variance if data has < 2 points after windowing
        if len(self.data) < 2:
             return 0 # Cannot compute meaningful variance/entropy with one or no data point
        avg = sum(self.data)/len(self.data)
        variance_sum = sum((x-avg)**2 for x in self.data)
        # Avoid division by zero if len(self.data) is 0 or 1 (handled above)
        # Avoid log(0) if variance is 0 (all data points are identical)
        variance = variance_sum / len(self.data)
        entropy = math.log(variance+1, 2) # Add 1 to variance to avoid log(0)
        return entropy


# ---------------------------------------------------------------------
# 9. CCGNPrototype
# ---------------------------------------------------------------------
class CCGNPrototype:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_relation(self, a, b, weight=1.0):
        self.graph.add_edge(a, b, weight=weight)

    def propagate(self):
        for a, b, d in self.graph.edges(data=True):
            d["weight"] *= 0.95
        return self.graph

# ---------------------------------------------------------------------
# 10. ForecastAgent
# ---------------------------------------------------------------------
class ForecastAgent:
    def __init__(self):
        self.history = []

    def forecast(self, entropy_values):
        if np is None or len(entropy_values) < 2:
            # Fallback if numpy is not available or not enough data for polyfit
            if not entropy_values:
                trend = 0
            else:
                # Simple trend based on last two points if available, otherwise 0
                if len(entropy_values) >= 2:
                     trend = entropy_values[-1] - entropy_values[-2]
                else:
                     trend = 0
        else:
            # Use numpy polyfit if available and enough data
            try:
                 # Suppress polyfit warnings about rank deficiency for small datasets
                 with warnings.catch_warnings():
                     warnings.simplefilter("ignore", np.RankWarning)
                     trend = np.polyfit(range(len(entropy_values)), entropy_values, 1)[0]
            except Exception as e:
                 print(f"Warning: Error during numpy polyfit: {e}. Falling back to simple trend.")
                 if len(entropy_values) >= 2:
                      trend = entropy_values[-1] - entropy_values[-2]
                 else:
                      trend = 0


        prediction = "rising" if trend > 0 else ("falling" if trend < 0 else "stable") # Add stable state
        self.history.append({"trend": trend, "prediction": prediction})
        return prediction

# ---------------------------------------------------------------------
# 11. ZKProofEngine
# ---------------------------------------------------------------------
class ZKProofEngine:
    def __init__(self):
        self.commits = []

    def commit(self, data: str):
        h = hashlib.sha256(data.encode()).hexdigest()
        self.commits.append(h)
        return h

    def verify(self, data: str, h: str):
        return hashlib.sha256(data.encode()).hexdigest() == h

# ---------------------------------------------------------------------
# 12. CCGNVisualizer
# ---------------------------------------------------------------------
class CCGNVisualizer:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def render(self):
        try:
            import matplotlib.pyplot as plt
            pos = nx.spring_layout(self.graph)
            weights = [self.graph[u][v]['weight'] for u,v in self.graph.edges()]
            # Add labels to edges based on weight if desired
            edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in self.graph.edges(data=True)}

            nx.draw(self.graph, pos, with_labels=True, width=weights, node_color='lightblue', node_size=700, font_size=10)
            # Draw edge labels
            nx.draw_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='red')

            plt.title("CCGN Visualization")
            plt.show()
        except ImportError:
            print("Visualization skipped: matplotlib not installed. Install with 'pip install matplotlib'.")
        except Exception as e:
            print("Visualization skipped:", e)

# ---------------------------------------------------------------------
# Demo Menu
# ---------------------------------------------------------------------
import warnings # Import warnings for numpy polyfit
if __name__ == "__main__":
    print("=== Tessrax Colab Full Runtime Demo ===")
    print("1: PredictiveRuntime\n2: IntegrityMonitor\n3: ZKAudit\n4: ClosureLedger\n5: EntropyForecaster\n"
          "6: LedgerIntegrityEngine\n7: GovernanceKernelRuntime\n8: EntropySignalAnalyzer\n9: CCGNPrototype\n"
          "10: ForecastAgent\n11: ZKProofEngine\n12: CCGNVisualizer")
    choice = input("Select module number to run demo (or ENTER to exit): ").strip()

    if choice == "1":
        rt = PredictiveRuntime()
        print("\n--- PredictiveRuntime Demo ---")
        test_values = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
        print("Input -> Prediction")
        for v in test_values:
            print(f"{v} -> {rt.run_cycle(v):.4f}")
        print("\nHistory:", rt.history)

    elif choice == "2":
        print("\n--- IntegrityMonitor Demo ---")
        # Create dummy files for demonstration
        dummy_dir = "temp_integrity_check"
        os.makedirs(dummy_dir, exist_ok=True)
        with open(os.path.join(dummy_dir, "file1.txt"), "w") as f:
            f.write("This is file one.")
        with open(os.path.join(dummy_dir, "file2.txt"), "w") as f:
            f.write("This is file two.")

        monitor = IntegrityMonitor(dummy_dir)
        report = monitor.verify_directory()
        print(f"\nIntegrity Report for '{dummy_dir}':")
        for filepath, filehash in report.items():
            print(f"  {filepath}: {filehash}")

        # Clean up dummy directory
        try:
             os.remove(os.path.join(dummy_dir, "file1.txt"))
             os.remove(os.path.join(dummy_dir, "file2.txt"))
             os.rmdir(dummy_dir)
             print(f"\nCleaned up dummy directory '{dummy_dir}'.")
        except Exception as e:
             print(f"Warning: Could not clean up dummy directory: {e}")


    elif choice == "3":
        print("\n--- ZKAudit Demo ---")
        # Use a fixed secret key for demo
        secret = b'supersecretkey'
        zka = ZKAudit(secret)
        message1 = "Transaction ABC occurred."
        message2 = "Transaction XYZ occurred."
        message3 = "Transaction ABC occurred." # Same message as message1

        commit1 = zka.commit(message1)
        commit2 = zka.commit(message2)
        commit3 = zka.commit(message3)

        print("\nCommit 1 (Message 1):", json.dumps(commit1, indent=2))
        print("\nCommit 2 (Message 2):", json.dumps(commit2, indent=2))
        print("\nCommit 3 (Message 3 - same as Message 1):", json.dumps(commit3, indent=2))

        # Verify if commit hashes match for identical messages
        print(f"\nCommit hash for Message 1 == Commit hash for Message 3: {commit1['message_hash'] == commit3['message_hash']}")
        print(f"Commit hash for Message 1 == Commit hash for Message 2: {commit1['message_hash'] == commit2['message_hash']}")


    elif choice == "4":
        print("\n--- ClosureLedger Demo ---")
        cl = ClosureLedger()

        # Add events with causal links
        cl.add_event("DataIngested", [])
        cl.add_event("ContradictionDetected", ["DataIngested"])
        cl.add_event("ResolutionProposed", ["ContradictionDetected"])
        cl.add_event("AmendmentApplied", ["ResolutionProposed"])
        cl.add_event("UnrelatedEvent", []) # Unrelated event


        print("\nInitial Ledger State (before closing):")
        print(cl.export_json())

        # Close some events
        cl.close_event("ContradictionDetected", {"status": "resolved", "details": "Manual review"})
        cl.close_event("AmendmentApplied", {"status": "complete"})

        print("\nLedger State after closing events:")
        print(cl.export_json())

        # Try to close a non-existent event
        try:
            cl.close_event("FakeEvent", {"status": "failed"})
        except ValueError as e:
            print(f"\nAttempted to close a fake event: {e}")


    elif choice == "5":
        print("\n--- EntropyForecaster Demo ---")
        ef = EntropyForecaster()

        print("Initial prediction:", ef.predict_next()) # Should be 0

        print("Updating with 0.1:", ef.update(0.1)) # Should be 0.1 (just return last)
        print("Updating with 0.3:", ef.update(0.3)) # Should be 0.5 (0.3 + (0.3-0.1))
        print("Updating with 0.2:", ef.update(0.2)) # Should be 0.1 (0.2 + ((0.3-0.1)+(0.2-0.3))/2) -> 0.2 + (0.2 - 0.1)/2 = 0.2 + 0.05 = 0.25?
        # Re-calculating: values=[0.1, 0.3, 0.2]. diffs=[0.2, -0.1]. avg_delta = (0.2 - 0.1)/2 = 0.05. Prediction = 0.2 + 0.05 = 0.25
        print("Updating with 0.2 (Corrected):", ef.update(0.2))

        print("\nHistory:", ef.values)
        print("Next predicted value:", ef.predict_next()) # Based on current history


    elif choice == "6":
        print("\n--- LedgerIntegrityEngine Demo ---")
        le = LedgerIntegrityEngine()

        entry1 = {"event_type": "DATA_POINT", "data": {"value": 100}}
        entry2 = {"event_type": "ALERT", "data": {"message": "High value detected"}}
        entry3 = {"event_type": "SYSTEM_EVENT", "data": {"status": "OK"}}

        hash1 = le.append_entry(entry1)
        hash2 = le.append_entry(entry2)
        hash3 = le.append_entry(entry3)

        print("\nAppended entries and got hashes:")
        print("Entry 1 Hash:", hash1)
        print("Entry 2 Hash:", hash2)
        print("Entry 3 Hash:", hash3)

        print("\nVerifying ledger integrity...")
        is_valid = le.verify_ledger()
        print("Ledger valid?", is_valid)

        # Simulate tampering (modify an entry in the list directly - not how a real ledger would be tampered)
        if le.ledger:
             print("\nSimulating ledger tampering (modifying data in memory)...")
             le.ledger[1]["entry"]["data"]["message"] = "Tampered message!"
             print("Verifying ledger integrity after tampering...")
             is_valid_tampered = le.verify_ledger()
             print("Ledger valid after tampering?", is_valid_tampered)
        else:
             print("\nSkipping tampering demo: Ledger is empty.")


    elif choice == "7":
        print("\n--- GovernanceKernelRuntime Demo ---")
        kernel = GovernanceKernelRuntime()

        print("\nSignal Strength -> Contradiction Detected")
        signals = [0.6, 0.8, 0.5, 0.9, 0.7]
        for signal in signals:
            contradiction = kernel.detect_contradiction(signal)
            # print(f"{signal:.2f} -> {contradiction}") # Output is already logged by the kernel's logger

        print("\nKernel Events:", kernel.events)


    elif choice == "8":
        print("\n--- EntropySignalAnalyzer Demo ---")
        analyzer = EntropySignalAnalyzer(window=3)

        print("Pushing 1.0 -> Entropy:", analyzer.push(1.0)) # Entropy 0 (one data point)
        print("Pushing 1.1 -> Entropy:", analyzer.push(1.1)) # Entropy > 0 (two data points, variance > 0)
        print("Pushing 1.2 -> Entropy:", analyzer.push(1.2)) # Entropy updated (three data points)
        print("Pushing 1.0 -> Entropy:", analyzer.push(1.0)) # Window full, oldest (1.0) removed, new 1.0 added [1.1, 1.2, 1.0]
        print("Pushing 1.5 -> Entropy:", analyzer.push(1.5)) # Window full, oldest (1.1) removed [1.2, 1.0, 1.5]

        print("\nAnalyzer Data Window:", analyzer.data)
        print("Current Entropy:", analyzer.compute_entropy())


    elif choice == "9":
        print("\n--- CCGNPrototype Demo ---")
        ccgn = CCGNPrototype()

        # Add relations
        ccgn.add_relation("Data Source A", "Claim 1", weight=0.8)
        ccgn.add_relation("Data Source B", "Claim 1", weight=0.7) # Conflict?
        ccgn.add_relation("Claim 1", "Contradiction 1", weight=0.9)
        ccgn.add_relation("Contradiction 1", "Resolution 1", weight=0.6)
        ccgn.add_relation("Resolution 1", "Policy Update", weight=0.5)
        ccgn.add_relation("Data Source C", "Claim 2", weight=0.95) # Unrelated chain

        print("\nInitial Graph Edges with Weights:")
        print(ccgn.graph.edges(data=True))

        print("\nPropagating influence (weights decay)...")
        ccgn.propagate() # First propagation
        ccgn.propagate() # Second propagation
        ccgn.propagate() # Third propagation

        print("\nGraph Edges with Weights after Propagation:")
        print(ccgn.graph.edges(data=True))


    elif choice == "10":
        print("\n--- ForecastAgent Demo ---")
        agent = ForecastAgent()

        print("\nForecasting with empty history:")
        print("Prediction:", agent.forecast([])) # Should be stable/0 trend

        print("\nForecasting with minimal history [0.1]:")
        print("Prediction:", agent.forecast([0.1])) # Should be stable/0 trend

        print("\nForecasting with rising trend [0.1, 0.3, 0.5]:")
        print("Prediction:", agent.forecast([0.1, 0.3, 0.5])) # Should be rising

        print("\nForecasting with falling trend [0.5, 0.3, 0.1]:")
        print("Prediction:", agent.forecast([0.5, 0.3, 0.1])) # Should be falling

        print("\nForecasting with mixed trend [0.1, 0.5, 0.2]:")
        print("Prediction:", agent.forecast([0.1, 0.5, 0.2])) # Should be falling (0.5->0.2 is strong fall)

        print("\nAgent History:", agent.history)


    elif choice == "11":
        print("\n--- ZKProofEngine Demo ---")
        zk = ZKProofEngine()

        data1 = "Sensitive data point A: value is 123.45"
        data2 = "Sensitive data point B: value is 678.90"
        data3 = "Sensitive data point A: value is 123.45" # Same as data1

        commit1 = zk.commit(data1)
        commit2 = zk.commit(data2)
        commit3 = zk.commit(data3)

        print("\nCommit 1 (Data 1 Hash):", commit1)
        print("Commit 2 (Data 2 Hash):", commit2)
        print("Commit 3 (Data 3 Hash):", commit3)

        print("\nVerifying Data 1 against Commit 1:", zk.verify(data1, commit1))
        print("Verifying Data 2 against Commit 1:", zk.verify(data2, commit1)) # Should be False
        print("Verifying Data 3 against Commit 1:", zk.verify(data3, commit1)) # Should be True (same data)

        # Simulate providing incorrect data for verification
        print("Verifying Tampered Data against Commit 1:", zk.verify("Tampered data", commit1)) # Should be False


    elif choice == "12":
        print("\n--- CCGNVisualizer Demo ---")
        # Create a sample graph for visualization
        graph_to_viz = nx.DiGraph()
        graph_to_viz.add_edge("Source A", "Claim 1", weight=1.0)
        graph_to_viz.add_edge("Source B", "Claim 1", weight=0.8)
        graph_to_viz.add_edge("Claim 1", "Contradiction 1", weight=1.0)
        graph_to_viz.add_edge("Contradiction 1", "Resolution 1", weight=0.7)
        graph_to_viz.add_edge("Claim 2", "Contradiction 1", weight=0.9) # Another source for contradiction
        graph_to_viz.add_edge("Resolution 1", "Amendment", weight=0.6)


        viz = CCGNVisualizer(graph_to_viz)
        print("\nRendering CCGN Visualization...")
        viz.render()
        print("Visualization window should appear (if matplotlib is installed and in a compatible environment).")


    else:
        print("\nNo module selected or invalid choice. Exiting demo.")


# Remove the usage instructions from the executed code block
# This ensures only the code runs when the cell is executed directly.
# The instructions are helpful in the markdown explanation.
# print("\n⸻")
# print("\n✅ Usage:")
# print("        1.      Save this block as tessrax_colab_full.py.")
# print("        2.      Install networkx (and optionally matplotlib, numpy):")
# print("\npip install networkx matplotlib numpy\n")
# print("        3.      Run:")
# print("\npython tessrax_colab_full.py\n")
# print("        4.      Choose a module number to test each subsystem interactively.")
# print("\nThis single file now executes every functional component from both Colab code bases.")
