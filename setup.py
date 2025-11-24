"""Setup script for efficiency_of_reason package."""

from setuptools import setup, find_packages
from pathlib import Path

# Get the directory containing this file
here = Path(__file__).parent
src_dir = here / "src"

# Verify src directory exists
if not src_dir.exists():
    raise RuntimeError(f"Source directory not found: {src_dir}")

# Find all packages in src directory
packages = find_packages(where=str(src_dir))

# Verify all expected packages are found
expected_packages = {"wor", "wor.core", "wor.data", "wor.metrics"}
found_packages = set(packages)
missing = expected_packages - found_packages
if missing:
    raise RuntimeError(f"Missing packages: {missing}. Found: {found_packages}")

setup(
    name="efficiency-of-reason",
    version="0.1.0",
    description="Quantifying the Computational Sparsity of Chain-of-Thought in Large Language Models",
    author="Emre Aktas",
    packages=packages,
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.24.0",
        "tabulate>=0.9.0",
    ],
    include_package_data=True,
    zip_safe=False,
)

