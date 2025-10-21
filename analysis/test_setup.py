#!/usr/bin/env python3
"""
Quick validation test for pipeline dependencies and setup.
Run this before executing the full pipeline.
"""

import sys
print("Testing Contradiction Energy Model Pipeline Setup")
print("="*60)

# Test 1: Python version
print("\n[1/6] Checking Python version...")
if sys.version_info >= (3, 8):
    print(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor} (OK)")
else:
    print(f"  ✗ Python {sys.version_info.major}.{sys.version_info.minor} (Need >= 3.8)")
    sys.exit(1)

# Test 2: Core dependencies
print("\n[2/6] Checking core dependencies...")
required = ['numpy', 'pandas', 'sklearn', 'scipy']
missing = []

for package in required:
    try:
        __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} (MISSING)")
        missing.append(package)

# Test 3: NLP dependencies
print("\n[3/6] Checking NLP dependencies...")
nlp_packages = ['sentence_transformers']

for package in nlp_packages:
    try:
        __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} (MISSING - install with: pip install sentence-transformers)")
        missing.append(package)

# Test 4: Statistics
print("\n[4/6] Checking statistics packages...")
stat_packages = ['statsmodels']

for package in stat_packages:
    try:
        __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} (MISSING)")
        missing.append(package)

# Test 5: Visualization
print("\n[5/6] Checking visualization packages...")
viz_packages = ['matplotlib', 'seaborn', 'plotly']

for package in viz_packages:
    try:
        __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} (MISSING)")
        missing.append(package)

# Test 6: ConvoKit (optional, will download on first run)
print("\n[6/6] Checking ConvoKit...")
try:
    import convokit
    print(f"  ✓ convokit")
except ImportError:
    print(f"  ! convokit (will be installed on first pipeline run)")

# Summary
print("\n" + "="*60)
if missing:
    print(f"SETUP INCOMPLETE: {len(missing)} packages missing")
    print("\nInstall missing packages with:")
    print(f"  pip install {' '.join(missing)}")
    print("\nOr install all at once:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
else:
    print("✓ ALL DEPENDENCIES INSTALLED")
    print("\nReady to run pipeline:")
    print("  python run_complete_pipeline.py")
    print("\nFor faster subsequent runs:")
    print("  python run_complete_pipeline.py --use-cache")

print("="*60)
