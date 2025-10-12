"""
Tessrax Demo Dispatcher
-----------------------
Run any demo from the unified registry:
    python demo/run_demo.py integration
    python demo/run_demo.py engine
    python demo/run_demo.py pledge_audit
    python demo/run_demo.py tmp1
"""

import sys, subprocess, json, pathlib

# === Registry ===============================================================
REGISTRY = {
    "integration": "demo/integration_demo.py",
    "engine": "demo/test_engine_demo.py",
    "pledge_audit": "demo/contradiction_demo2.py",
    "tmp1": "demo/tmp1.py",
    "governance": "demo/agent_governance_demo.py",
    "agent": "demo/agent_demo.py",
}

# === Entry Point ============================================================
def main():
    if len(sys.argv) < 2:
        print("\nUsage: python demo/run_demo.py <demo_name>\n")
        print("Available demos:")
        for k in REGISTRY:
            print(f"  • {k}")
        print()
        sys.exit(1)

    demo_name = sys.argv[1]
    if demo_name not in REGISTRY:
        print(f"❌ Unknown demo: {demo_name}")
        print("Valid options:", ", ".join(REGISTRY.keys()))
        sys.exit(1)

    path = pathlib.Path(REGISTRY[demo_name])
    if not path.exists():
        print(f"❌ Demo file not found: {path}")
        sys.exit(1)

    print(f"\n🚀 Running Tessrax demo: {demo_name}\n{'-'*60}")
    try:
        subprocess.run([sys.executable, str(path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n💥 Demo {demo_name} failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user.")
        sys.exit(130)

    print(f"\n✅ Demo '{demo_name}' completed successfully.\n")


if __name__ == "__main__":
    main()