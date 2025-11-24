# Troubleshooting Guide

## Memory Issues (`std::bad_alloc`)

**CRITICAL:** The `std::bad_alloc` error means your system ran out of RAM during model loading.

### Understanding the Problem

When loading DeepSeek-R1-Distill-Llama-8B with 4-bit quantization:
- The model requires ~16GB VRAM (GPU memory) after quantization
- **BUT**: During loading, it needs significant CPU RAM for initialization
- The bitsandbytes library loads the full model to CPU RAM first, THEN quantizes it
- This creates a memory spike that can exceed your container limits

### Check Your Memory Limits

If you're running in a Docker container (like RunPod), check for **cgroup memory limits**:

```bash
python scripts/check_system_limits.py
```

Even if your host system has 251GB RAM, the container might be limited to ~38GB by cgroups.

### Solutions (In Order of Preference)

#### Quick fix for `std::bad_alloc` on 1.5B models

- The alternative loader now uses a safer streaming path with explicit CPU/GPU limits.
- If your host has plenty of RAM but cgroup detection is misleading, bypass it:

```bash
export SKIP_CGROUP_LIMIT_CHECK=true   # ignore the 38GB cgroup file
export USE_ALTERNATIVE_LOADER=true    # use the safer loader
python scripts/01_experiment_sparsity_gap_enhanced.py --num-problems 5
```

If the direct FP16 stream still fails, set `USE_8BIT_QUANTIZATION=true` to force the fallback.

#### Solution 1: Use 8-bit Quantization (RECOMMENDED for 32-48GB RAM containers)

8-bit quantization is more reliable than 4-bit for memory-constrained environments:

```bash
export USE_ALTERNATIVE_LOADER=true
export USE_8BIT_QUANTIZATION=true
python scripts/01_run_sparsity_gap.py
```

**Trade-offs:**
- ✅ More reliable loading
- ✅ Works with 32-48GB RAM
- ⚠️ Uses slightly more VRAM (~20GB vs ~16GB)
- ⚠️ Slightly slower inference

#### Solution 2: Upgrade to a Larger RunPod Instance

For the best experience with 4-bit quantization:
- **Minimum for 4-bit:** 48GB RAM
- **Recommended:** 64GB+ RAM
- After upgrading, the default loader should work without modifications

#### Solution 3: Disable Memory Limits (USE WITH CAUTION)

If you have 48GB+ RAM and still get errors, the automatic limits may be too aggressive:

```bash
export DISABLE_ALL_MEMORY_LIMITS=true
python scripts/01_run_sparsity_gap.py
```

**⚠️ WARNING:** This may cause system instability if you actually don't have enough RAM.

#### Solution 4: Load Without Quantization (Requires High VRAM)

If you have a GPU with 32GB+ VRAM (e.g., A100):

```bash
export USE_ALTERNATIVE_LOADER=true
python scripts/01_run_sparsity_gap.py
```

This loads the model in FP16 without quantization.

### Updated Memory Requirements Table

| Configuration | GPU VRAM | CPU RAM | RunPod Instance |
|--------------|----------|---------|-----------------|
| 4-bit quantization | ~16GB | 48GB+ | RTX 4090 (48GB RAM) |
| 8-bit quantization | ~20GB | 32GB+ | RTX 4090 (32GB RAM) |
| No quantization (FP16) | ~32GB | 16GB+ | A100 (40GB VRAM) |

### Technical Details

The loader now automatically detects cgroup limits and adjusts memory allocation:
- For containers with <48GB RAM: Uses only 25% of RAM for CPU (forces more disk offloading)
- For containers with 48GB+ RAM: Uses 80% of RAM for CPU (standard behavior)
- GPU memory is limited to 85% of available VRAM

If you're still having issues after trying these solutions, check:

```bash
free -h  # Check available RAM
nvidia-smi  # Check GPU VRAM
cat /sys/fs/cgroup/memory/memory.limit_in_bytes  # Check container limit
```

## Import Errors

If you get `ModuleNotFoundError: No module named 'wor.data'`:

1. Make sure you've pulled the latest changes: `git pull origin main`
2. Reinstall the package: `pip install -e .`
3. Verify: `python test_imports.py`

## Old Project Interference

If you see imports from `/workspace/Weight_of_Reasoning`:

1. Run the cleanup script: `bash cleanup_old_project.sh`
2. Reinstall: `pip install -e .`

## CUDA/GPU Issues

If the model doesn't use GPU:

1. Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
2. Verify GPU: `nvidia-smi`
3. Make sure you're using a GPU-enabled RunPod instance
