#!/bin/bash
# Fix installation script for RunPod

echo "=== Fixing efficiency-of-reason installation ==="

# Uninstall existing package
echo "1. Uninstalling existing package..."
pip uninstall -y efficiency-of-reason 2>/dev/null || true

# Clear pip cache
echo "2. Clearing pip cache..."
pip cache purge 2>/dev/null || true

# Reinstall
echo "3. Reinstalling package..."
pip install -e .

# Verify installation
echo "4. Verifying installation..."
python test_imports.py

echo "=== Done! ==="

