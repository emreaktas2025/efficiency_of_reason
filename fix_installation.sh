#!/bin/bash
# Fix installation script for RunPod

echo "=== Fixing efficiency-of-reason installation ==="

# Clean up old project interference
echo "0. Cleaning up old project interference..."
if [ -f "/usr/local/lib/python3.12/dist-packages/_weight_of_reasoning.pth" ]; then
    rm -f /usr/local/lib/python3.12/dist-packages/_weight_of_reasoning.pth 2>/dev/null || \
    sudo rm -f /usr/local/lib/python3.12/dist-packages/_weight_of_reasoning.pth 2>/dev/null || true
    echo "   Removed old .pth file"
fi

# Uninstall existing package
echo "1. Uninstalling existing package..."
pip uninstall -y efficiency-of-reason 2>/dev/null || true
pip uninstall -y weight-of-reasoning 2>/dev/null || true

# Remove any .egg-info or build directories
echo "2. Cleaning build artifacts..."
rm -rf *.egg-info src/*.egg-info build dist 2>/dev/null || true

# Clear pip cache
echo "3. Clearing pip cache..."
pip cache purge 2>/dev/null || true

# Verify directory structure
echo "4. Verifying directory structure..."
if [ ! -d "src/wor/data" ]; then
    echo "   ERROR: src/wor/data directory not found!"
    echo "   Current directory: $(pwd)"
    echo "   Contents of src/wor: $(ls -la src/wor/ 2>/dev/null || echo 'src/wor not found')"
    exit 1
fi
echo "   Directory structure OK"

# Reinstall
echo "5. Reinstalling package..."
pip install --no-cache-dir -e .

# Verify installation
echo "6. Verifying installation..."
python test_imports.py

echo "=== Done! ==="

