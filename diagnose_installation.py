#!/usr/bin/env python3
"""Diagnose package installation issues."""

import sys
import os
from pathlib import Path

print("=== Package Installation Diagnostics ===\n")

# Check if package is installed
try:
    import wor
    print(f"✓ wor module found at: {wor.__file__}")
    print(f"  wor.__path__: {wor.__path__}")
except ImportError as e:
    print(f"✗ wor module not found: {e}")
    sys.exit(1)

# Check subpackages
subpackages = ['core', 'data', 'metrics']
for subpkg in subpackages:
    try:
        mod = __import__(f'wor.{subpkg}', fromlist=[''])
        print(f"✓ wor.{subpkg} found at: {mod.__file__}")
    except ImportError as e:
        print(f"✗ wor.{subpkg} NOT found: {e}")

# Check sys.path
print(f"\n=== sys.path ===")
for i, path in enumerate(sys.path):
    marker = " <-- project src" if 'efficiency_of_reason' in path or 'wor' in path else ""
    print(f"{i}: {path}{marker}")

# Check installed package location
print(f"\n=== Package Location ===")
try:
    import site
    site_packages = site.getsitepackages()
    print(f"Site packages: {site_packages}")
    
    # Look for .pth files
    for sp in site_packages:
        pth_files = list(Path(sp).glob('*.pth'))
        if pth_files:
            print(f"\nFound .pth files in {sp}:")
            for pth in pth_files:
                print(f"  {pth.name}")
                if 'efficiency' in pth.name or 'wor' in pth.name:
                    print(f"    Contents: {pth.read_text()}")
except Exception as e:
    print(f"Could not check site packages: {e}")

# Check for .egg-link
print(f"\n=== .egg-link files ===")
try:
    import site
    for sp in site.getsitepackages():
        egg_links = list(Path(sp).glob('*.egg-link'))
        for el in egg_links:
            if 'efficiency' in el.name:
                print(f"Found: {el}")
                print(f"  Points to: {el.read_text().strip()}")
except Exception as e:
    print(f"Could not check .egg-link files: {e}")

print("\n=== Done ===")

