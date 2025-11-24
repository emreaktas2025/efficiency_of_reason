# Troubleshooting Guide

## Memory Issues (`std::bad_alloc`)

If you encounter `std::bad_alloc` errors when loading the model, this means the system ran out of RAM. The loader now includes automatic memory limits, but if you still have issues:

### Solution 1: Reduce CPU Memory Limit

Edit `src/wor/core/loader.py` and reduce the CPU memory limit:

```python
max_memory["cpu"] = "4GiB"  # Reduce from 8GiB to 4GiB
```

### Solution 2: Use a Larger RunPod Instance

- Minimum recommended: 16GB RAM
- Recommended: 32GB+ RAM for smoother operation

### Solution 3: Pre-download Model

Download the model first to avoid memory spikes during download:

```python
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    cache_dir="./models"
)
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

