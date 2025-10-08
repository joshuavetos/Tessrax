"""
Measures throughput and latency of contradiction detection.
"""
import time, random, json
from conflict_graph import ConflictGraph

sample = [{"text": f"Statement {i} value {random.randint(0,100)}"} for i in range(500)]
cg = ConflictGraph()

start = time.time()
cg.add_statements(sample)
cg.compute_edges()
elapsed = time.time() - start

print(json.dumps({
    "num_statements": len(sample),
    "num_edges": cg.graph.number_of_edges(),
    "runtime_seconds": round(elapsed, 2)
}, indent=2))