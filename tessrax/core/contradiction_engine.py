import itertools
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any
import networkx as nx
import time # Import the time module
import hashlib # Import the hashlib module
import json # Import the json module

# Define data structures and core logic for contradiction detection

class ClaimType(Enum):
    """Types of claims or statements."""
    FACT = "fact"
    NORMATIVE = "normative"
    PREDICTION = "prediction"
    OPINION = "opinion"
    UNKNOWN = "unknown"

@dataclass
class Claim:
    """Represents a single claim or statement."""
    agent: str          # Identifier of the agent making the claim (e.g., human, AI, system)
    claim: str          # The statement itself
    type: ClaimType     # Type of claim (e.g., FACT, NORMATIVE)
    timestamp: float = field(default_factory=lambda: time.time()) # Timestamp of the claim
    id: str = field(default_factory=lambda: hashlib.sha256(str(time.time()).encode()).hexdigest()) # Unique ID

    def __hash__(self):
        # Make Claim objects hashable for networkx nodes
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Claim):
            return NotImplemented
        return self.id == other.id

def detect_contradictions(claims: List[Dict[str, Any]]) -> nx.Graph:
    """
    Detects contradictions within a list of claims.

    Args:
        claims: A list of dictionaries, where each dictionary represents a claim.
                Expected keys: 'agent', 'claim', 'type' (optional, defaults to UNKNOWN).

    Returns:
        A networkx Graph where nodes are claims and edges represent contradictions.
    """
    G = nx.Graph()
    claim_objects = []

    # Convert dictionaries to Claim objects and add as nodes
    for claim_data in claims:
        try:
            claim_type = ClaimType(claim_data.get("type", "unknown").lower())
        except ValueError:
            claim_type = ClaimType.UNKNOWN

        # Create a Claim object, handling potential missing fields gracefully
        # Assign a unique ID if not provided (though default_factory should handle this)
        # Ensure required fields are present or handle their absence
        if 'agent' not in claim_data or 'claim' not in claim_data:
             print(f"Warning: Skipping claim with missing agent or claim field: {claim_data}")
             continue

        claim_obj = Claim(
            agent=claim_data['agent'],
            claim=claim_data['claim'],
            type=claim_type,
            # timestamp and id are handled by default_factory, but can be overridden if present
            timestamp=claim_data.get('timestamp', time.time()),
            id=claim_data.get('id', hashlib.sha256(str(time.time()).encode()).hexdigest())
        )
        claim_objects.append(claim_obj)
        G.add_node(claim_obj)

    # Simple pairwise contradiction detection (can be replaced with more sophisticated logic)
    # This is a placeholder; real contradiction detection is complex and domain-specific.
    for claim1, claim2 in itertools.combinations(claim_objects, 2):
        # Example basic contradiction check:
        # If two normative claims from different agents are exact opposites.
        if claim1.type == ClaimType.NORMATIVE and claim2.type == ClaimType.NORMATIVE:
             # A very naive check: check for explicit negation or antonyms.
             # This needs sophisticated NLP for real use.
             # Placeholder: check if one claim contains "not" and the other doesn't, and they are otherwise similar.
             # Or if keywords suggest opposition (e.g., "allow" vs "deny").
             # For this example, let's just assume any two distinct normative claims from different agents *could* be in conflict
             # requiring governance review, though not necessarily a logical contradiction.
             # A true contradiction requires semantic analysis.

             # Let's implement a slightly less naive check: look for simple negation patterns.
             claim1_lower = claim1.claim.lower()
             claim2_lower = claim2.claim.lower()

             # Simple negation check (very basic)
             if ((" not " in claim1_lower and " not " not in claim2_lower and claim1_lower.replace(" not ", " ") == claim2_lower) or
                 (" not " in claim2_lower and " not " not in claim1_lower and claim2_lower.replace(" not ", " ") == claim1_lower)):
                  G.add_edge(claim1, claim2, type="negation_contradiction")
                  print(f"Detected potential negation contradiction between '{claim1.claim}' and '{claim2.claim}'")
             # Add more sophisticated checks here (e.g., keyword antonyms, semantic similarity analysis)

        # Example: Fact vs. Fact contradiction (e.g., conflicting data points)
        if claim1.type == ClaimType.FACT and claim2.type == ClaimType.FACT:
             # This would require comparing the underlying data or semantic content.
             # Placeholder: a very basic check for exact opposite numerical values if the claim contains numbers.
             pass # Needs actual data comparison logic

        # Example: Fact vs. Prediction contradiction
        if claim1.type == ClaimType.FACT and claim2.type == ClaimType.PREDICTION:
            # Check if the fact contradicts the prediction outcome.
            pass # Needs logic to compare fact to prediction

        # Add other cross-type contradiction checks as needed

    return G

def score_stability(G: nx.Graph) -> float:
    """
    Calculates a stability score based on the contradiction graph.

    A higher score indicates fewer contradictions or less impact from contradictions.
    This is a simplified example; real stability scoring can be complex.

    Args:
        G: The contradiction graph.

    Returns:
        A float between 0 and 1, where 1 is perfectly stable (no contradictions).
    """
    total_possible_edges = len(G.nodes()) * (len(G.nodes()) - 1) / 2
    if total_possible_edges == 0:
        return 1.0 # Perfectly stable if no nodes
    num_contradiction_edges = G.number_of_edges()

    # Simple inverse relationship with number of contradictions
    # Could be weighted by centrality of nodes in contradiction graph, etc.
    stability = max(0.0, 1.0 - (num_contradiction_edges / total_possible_edges))
    return stability

def log_to_ledger(G: nx.Graph, stability: float, ledger_path: str | Path = Path("data/governance_ledger.jsonl")):
    """
    Logs detected contradictions and stability score to a ledger file.

    Args:
        G: The contradiction graph.
        stability: The calculated stability score.
        ledger_path: Path to the ledger file.
    """
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True) # Ensure data directory exists

    event = {
        "timestamp": time.time(),
        "type": "contradiction_report",
        "payload": {
            "contradictions": [{"source": edge[0].id, "target": edge[1].id, "type": edge[2].get('type', 'unknown')} for edge in G.edges(data=True)],
            "stability_score": stability,
            "num_claims": G.number_of_nodes(),
            "num_contradictions": G.number_of_edges(),
        }
    }

    # Simple append-only log (requires external mechanism for integrity checks like hashing)
    try:
        with open(ledger_path, "a") as f:
            json.dump(event, f)
            f.write('\n') # Add newline for each event
        print(f"Logged contradiction report to {ledger_path}")
    except Exception as e:
        print(f"Error logging to ledger: {e}")


# Example Usage (for demonstration, not typically run directly)
if __name__ == "__main__":
    print("Running example contradiction detection...")

    sample_claims = [
        {"agent": "Alice", "claim": "The sky is blue.", "type": "fact"},
        {"agent": "Bob", "claim": "The sky is green.", "type": "fact"},
        {"agent": "Charlie", "claim": "We must reduce carbon emissions.", "type": "normative"},
        {"agent": "David", "claim": "We must not reduce carbon emissions.", "type": "normative"},
        {"agent": "Alice", "claim": "The stock price will double tomorrow.", "type": "prediction"},
        {"agent": "Bob", "claim": "The stock price will halve tomorrow.", "type": "prediction"},
    ]

    contradiction_graph = detect_contradictions(sample_claims)
    current_stability = score_stability(contradiction_graph)

    print(f"\nDetected {contradiction_graph.number_of_edges()} contradictions.")
    print(f"Current stability score: {current_stability:.2f}")

    # Log to a temporary ledger file for this example run
    log_to_ledger(contradiction_graph, current_stability, "example_ledger.jsonl")

    print("\nContradiction Graph Edges:")
    for u, v, data in contradiction_graph.edges(data=True):
        print(f"- Contradiction between '{u.claim}' and '{v.claim}' (Type: {data.get('type', 'unknown')})")

    print("\nExample complete.")

"""
TESSRAX METABOLISM CORE — CONTRADICTION TENSOR (AUDITED REVISION)
-----------------------------------------------------------------
Implements a bounded, falsifiable, observer-dependent contradiction model.

Audit Enhancements (Oct 2025):
    • Normalized sim/opp → [0,1] domain for interpretability.
    • Volatility capped to prevent runaway entropy.
    • Gradient clipping and spectral normalization added for stability.
    • Symmetric field diffusion compatible with CE-MOD-66 lattice.
    • GovernanceKernel-ready event structure with deterministic hashes.

Each contradiction is treated as a localized tensor event:
    T(A, B, W) = f(sim01(A,B), opp01(A,B), κ·misalignment)
"""

import numpy as np
import hashlib


# === BASIC UTILITIES ===
def normalize(x):
    n = np.linalg.norm(x) + 1e-9
    return x / n


def sim01(A, B):
    """Dot similarity mapped to [0,1]."""
    return 0.5 * (np.clip(np.dot(A, B), -1.0, 1.0) + 1.0)


def opp01(A, B):
    """Opposition in [0,1]; complement of sim01."""
    return 1.0 - sim01(A, B)


def observer_misalignment(A, B, W):
    """Projection divergence along observer basis."""
    return abs(float(np.dot(A, W) - np.dot(B, W)))


def clip_by_norm(x, max_norm=3.0):
    """Prevent gradient explosions."""
    n = np.linalg.norm(x) + 1e-9
    return x * (max_norm / n) if n > max_norm else x


# === CORE CONTRADICTION TENSOR ===
def contradiction_tensor(A, B, W, kappa=0.5, c_thresh=0.8, alpha=0.8):
    """
    Computes scalar contradiction metrics:
        coherence ∈ [0,1]
        volatility ∈ [0,1]
        resolution_force ∈ [0,1]
        entropy_delta ∈ [-α,+α]
    """
    A, B, W = normalize(A), normalize(B), normalize(W)
    c = sim01(A, B)
    v = opp01(A, B) * (1.0 + kappa * observer_misalignment(A, B, W))
    v = min(1.0, v)
    r = max(0.0, c_thresh - c) * v
    dS = alpha * np.tanh(v - c)  # bounded ΔS
    return {
        "coherence": float(c),
        "volatility": float(v),
        "resolution_force": float(r),
        "entropy_delta": float(dS)
    }


def resolution_pressure(A, B, W, kappa=0.5):
    """Directional contradiction pressure vector (optional for CE-MOD-66)."""
    mis = observer_misalignment(A, B, W)
    dir_vec = normalize(B - A)
    mag = opp01(A, B) * (1.0 + kappa * mis)
    return mag * dir_vec


# === TEMPORAL METABOLISM LOOP ===
def step(A, B, W, eta=0.08, half_life=9.0, beta=1.0, gamma=0.6,
         momentum=0.8, mA=None, mB=None):
    """
    One metabolism cycle — tension evolves, resolves, or decays.
    """
    A, B, W = normalize(A), normalize(B), normalize(W)

    # Potential U = β(A·B) − γ|A·W − B·W|
    dU_dA = beta * B - gamma * np.sign(np.dot(A, W) - np.dot(B, W)) * W
    dU_dB = beta * A + gamma * np.sign(np.dot(A, W) - np.dot(B, W)) * W

    mA = momentum * (mA if mA is not None else np.zeros_like(A)) + dU_dA
    mB = momentum * (mB if mB is not None else np.zeros_like(B)) + dU_dB
    mA, mB = clip_by_norm(mA), clip_by_norm(mB)

    lam = np.log(2.0) / max(half_life, 1e-3)

    stable_dir = normalize(A + B)
    repairA = np.dot(A, stable_dir) * stable_dir
    repairB = np.dot(B, stable_dir) * stable_dir

    A_next = normalize(A + eta * mA - lam * (A - repairA))
    B_next = normalize(B + eta * mB - lam * (B - repairB))

    metrics = contradiction_tensor(A_next, B_next, W)
    return A_next, B_next, mA, mB, metrics


# === FIELD DYNAMICS (CE-MOD-66 COMPATIBLE) ===
def spectral_normalize(M, sigma=1.0):
    """Ensures ||M̂||₂ ≤ 1."""
    eigvals = np.linalg.eigvalsh(M)
    max_eig = max(abs(eigvals)) + 1e-9
    return (sigma / max(1.0, max_eig)) * M


def contradiction_tensor_field(nodes, observers, M=None):
    """
    Compute contradiction field over semantic lattice.
    Symmetric observer blending; optional spectral graph diffusion.
    """
    N = len(nodes)
    field = np.zeros((N, N, 4))
    for i in range(N):
        for j in range(N):
            W_ij = normalize(observers[i] + observers[j]) if observers is not None else np.ones_like(nodes[i])
            metrics = contradiction_tensor(nodes[i], nodes[j], W_ij)
            field[i, j] = np.array(list(metrics.values()))

    if M is not None:
        d = np.sum(M, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-6)))
        M_hat = spectral_normalize(D_inv_sqrt @ M @ D_inv_sqrt)
        # Symmetric diffusion: apply to both indices
        field = np.einsum('ij,jkl->ikl', M_hat, field)
        field = np.einsum('ij,kil->kjl', M_hat, field)

    return field


# === GOVERNANCE INTEGRATION ===
class GovernanceKernel:
    """Stub for receipt-level event logging."""
    def __init__(self):
        self.log = []

    def record(self, scard_id, metrics, A=None, B=None, W=None, params=None):
        proj = (
            float(np.dot(A, W)) if A is not None else 0.0,
            float(np.dot(B, W)) if B is not None else 0.0,
            float(np.dot(A, B)) if A is not None and B is not None else 0.0
        )
        payload = f"{proj}{params}".encode()
        digest = hashlib.sha256(payload).hexdigest()[:16]
        entry = {
            "scard_id": scard_id,
            "metrics": metrics,
            "hash": digest,
            "params": params or {}
        }
        self.log.append(entry)


# === TEST / FALSIFIABILITY HARNESS ===
def quick_tests():
    np.random.seed(0)
    A = normalize(np.random.randn(64))
    B = normalize(np.random.randn(64))
    W = normalize(np.random.randn(64))
    print("Baseline tensor:", contradiction_tensor(A, B, W))

    print("\nκ sweep (volatility monotonicity):")
    for k in np.linspace(0, 1, 6):
        v = contradiction_tensor(A, B, W, kappa=k)['volatility']
        print(f"kappa={k:.2f} → volatility={v:.3f}")

    # Stress test step loop
    A, B, W = normalize(A), normalize(B), normalize(W)
    for _ in range(5):
        A, B, *_ , m = step(A, B, W)
        print("cycle:", m)


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    np.random.seed(42)
    A = normalize(np.random.randn(128))
    B = normalize(np.random.randn(128))
    W = normalize(np.random.randn(128))
    gov = GovernanceKernel()

    for t in range(10):
        A, B, mA, mB, metrics = step(A, B, W)
        gov.record(f"scard-{t}", metrics, A, B, W, {"kappa": 0.5, "alpha": 0.8})
        print(f"Cycle {t:02d}:", metrics)

    print("\nGovernance log entries:", len(gov.log))
    quick_tests()