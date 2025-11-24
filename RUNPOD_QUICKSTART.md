# RunPod Quick Start Guide

## Setup Commands

After cloning the repository on RunPod, run these commands:

```bash
git clone https://github.com/emreaktas2025/efficiency_of_reason.git
cd efficiency_of_reason

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode (recommended)
pip install -e .
```

## Run Experiment

```bash
# Make sure you're in the project root
cd /workspace/efficiency_of_reason  # or wherever you cloned it

# Run the experiment
python scripts/01_run_sparsity_gap.py
```

## Alternative: Without Package Installation

If you prefer not to install the package:

```bash
# Just install dependencies
pip install -r requirements.txt

# Run from project root (script will handle path setup)
cd /workspace/efficiency_of_reason
python scripts/01_run_sparsity_gap.py
```

## Troubleshooting

**If you get `ModuleNotFoundError: No module named 'wor'`:**

1. Make sure you're in the project root directory
2. Try installing the package: `pip install -e .`
3. Or verify the `src/wor` directory exists

**If you get CUDA/GPU errors:**

- Make sure you're using a GPU-enabled RunPod instance
- Check that PyTorch can see the GPU: `python -c "import torch; print(torch.cuda.is_available())"`

