#!/bin/bash
# Fix installation script for RunPod

echo "=== Fixing efficiency-of-reason installation ==="

# Uninstall existing package
echo "1. Uninstalling existing package..."
pip uninstall -y efficiency-of-reason 2>/dev/null || true

# Remove any .egg-info or build directories
echo "2. Cleaning build artifacts..."
rm -rf *.egg-info build dist 2>/dev/null || true

# Clear pip cache
echo "3. Clearing pip cache..."
pip cache purge 2>/dev/null || true

# Reinstall
echo "4. Reinstalling package..."
pip install --no-cache-dir -e .

# Verify installation
echo "5. Verifying installation..."
python test_imports.py

echo "=== Done! ==="

