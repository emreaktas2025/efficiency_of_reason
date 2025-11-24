#!/usr/bin/env python3
"""Quick test to verify imports work correctly."""

import sys
from pathlib import Path

# Add src to path (same logic as main script)
script_dir = Path(__file__).resolve().parent
project_root = script_dir
src_path = project_root / "src"

if src_path.exists():
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    print(f"✓ Added {src_path} to sys.path")
else:
    print(f"✗ Could not find {src_path}")
    sys.exit(1)

# Test imports
try:
    from wor.core import load_deepseek_r1_model
    print("✓ wor.core imported successfully")
except ImportError as e:
    print(f"✗ Failed to import wor.core: {e}")
    sys.exit(1)

try:
    from wor.data import parse_reasoning_output, get_token_indices_for_segments
    print("✓ wor.data imported successfully")
except ImportError as e:
    print(f"✗ Failed to import wor.data: {e}")
    sys.exit(1)

try:
    from wor.metrics import calculate_metrics_for_segment
    print("✓ wor.metrics imported successfully")
except ImportError as e:
    print(f"✗ Failed to import wor.metrics: {e}")
    sys.exit(1)

print("\n✅ All imports successful! You're ready to run the experiment.")
print("\nTo run the experiment:")
print("  python scripts/01_run_sparsity_gap.py")
print("\nOr install the package for easier imports:")
print("  pip install -e .")

