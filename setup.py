"""Setup script for efficiency_of_reason package."""

from setuptools import setup

# Explicitly list all packages to ensure they're all included
setup(
    name="efficiency-of-reason",
    version="0.1.0",
    description="Quantifying the Computational Sparsity of Chain-of-Thought in Large Language Models",
    author="Emre Aktas",
    packages=[
        "wor",
        "wor.core",
        "wor.data",
        "wor.metrics",
    ],
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.24.0",
        "tabulate>=0.9.0",
    ],
    # Ensure all package data is included
    include_package_data=True,
    zip_safe=False,
)

