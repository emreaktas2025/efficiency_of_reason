# RunPod Quick Start Guide

## Setup Commands

After cloning the repository on RunPod, run these commands:

```bash
# Clone the repository
git clone https://github.com/emreaktas2025/efficiency_of_reason.git
cd efficiency_of_reason

# Pull latest changes (if you already cloned)
git pull origin main

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode (RECOMMENDED - fixes import issues)
pip install -e .

# Verify imports work
python test_imports.py
```

## Test Model Loading First (IMPORTANT!)

Before running experiments, test if model loading works on your instance:

```bash
# Test model loading with different strategies
python scripts/test_model_loading.py
```

This will automatically try different quantization strategies and tell you which one works best for your system.

## Run Experiment

### Option 1: Standard 4-bit Quantization (Default)

```bash
# Make sure you're in the project root
cd /workspace/efficiency_of_reason  # or wherever you cloned it

# Run the experiment
python scripts/01_run_sparsity_gap.py
```

**Note:** If you get a `std::bad_alloc` error, see the memory solutions below.

### Option 2: Use 8-bit Quantization (Recommended for 32-48GB RAM)

If 4-bit quantization fails with memory errors:

```bash
export USE_ALTERNATIVE_LOADER=true
export USE_8BIT_QUANTIZATION=true
python scripts/01_run_sparsity_gap.py
```

### Option 3: Disable Memory Limits (Use with Caution)

If you have 48GB+ RAM but still get errors:

```bash
export DISABLE_ALL_MEMORY_LIMITS=true
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

### Memory Errors (`std::bad_alloc`)

**Problem:** The model fails to load with a memory allocation error.

**Root Cause:** 4-bit quantization requires significant CPU RAM during initialization, even though the final model uses less GPU memory. Your container may have cgroup limits (e.g., 38GB) that are insufficient.

**Solutions (in order of preference):**

1. **Use 8-bit quantization** (works with 32-48GB RAM):
   ```bash
   export USE_ALTERNATIVE_LOADER=true
   export USE_8BIT_QUANTIZATION=true
   python scripts/01_run_sparsity_gap.py
   ```

2. **Upgrade to a larger RunPod instance** (48GB+ RAM recommended for 4-bit)

3. **Disable memory limits** (if you have 48GB+ but still get errors):
   ```bash
   export DISABLE_ALL_MEMORY_LIMITS=true
   python scripts/01_run_sparsity_gap.py
   ```

For more details, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### Import Errors

**If you get `ModuleNotFoundError: No module named 'wor'`:**

1. Make sure you're in the project root directory
2. Try installing the package: `pip install -e .`
3. Or verify the `src/wor` directory exists

### GPU Errors

**If you get CUDA/GPU errors:**

- Make sure you're using a GPU-enabled RunPod instance
- Check that PyTorch can see the GPU: `python -c "import torch; print(torch.cuda.is_available())"`

## Quick Reference

### Memory Requirements by Configuration

| Configuration | GPU VRAM | CPU RAM | Best For |
|--------------|----------|---------|----------|
| 4-bit quantization | ~16GB | 48GB+ | Best quality, needs more RAM |
| 8-bit quantization | ~20GB | 32GB+ | **Recommended for most setups** |
| No quantization (FP16) | ~32GB | 16GB+ | High VRAM GPUs (A100) |

### Useful Commands

```bash
# Check system resources
nvidia-smi                    # Check GPU VRAM
free -h                       # Check available RAM
python scripts/check_system_limits.py  # Check container limits

# Test model loading
python scripts/test_model_loading.py

# Pull latest code
git pull origin main

# Reinstall if needed
pip install -e .
```

