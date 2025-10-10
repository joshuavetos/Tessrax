"""
Demo: Tessrax Minimal Protocol (TMP-1)
Runs the universal contradiction-governance loop in real time.
"""

from Core.protocols.tmp1 import step, L

print("\n🧠 Tessrax Minimal Protocol — Live Demo\n")

claims = ["system stable", "not system stable", "performance high"]
for c in claims:
    record = step(c)
    print(f"→ {c}")
    print(f"   stability={record['stability']:.2f}, route={record['route']}")
    print(f"   state={record['state']}")
    print("   hash:", record['hash'][:16], "…\n")

print("✅ Ledger length:", len(L))
print("Demo complete.\n")