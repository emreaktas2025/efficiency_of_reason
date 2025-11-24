#!/bin/bash
# Cleanup script to remove old Weight_of_Reasoning project interference

echo "=== Cleaning up old project interference ==="

# Remove old .pth file
echo "1. Removing old .pth file..."
if [ -f "/usr/local/lib/python3.12/dist-packages/_weight_of_reasoning.pth" ]; then
    sudo rm -f /usr/local/lib/python3.12/dist-packages/_weight_of_reasoning.pth
    echo "   Removed _weight_of_reasoning.pth"
else
    echo "   _weight_of_reasoning.pth not found (may already be removed)"
fi

# Uninstall any old packages
echo "2. Uninstalling old packages..."
pip uninstall -y weight-of-reasoning 2>/dev/null || echo "   No old package found"

echo "=== Done! ==="
echo ""
echo "Now run: pip install -e ."

