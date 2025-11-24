"""
Model loader for DeepSeek-R1-Distill-Llama-8B with 4-bit quantization.

This module handles loading the model with memory-efficient settings
for single GPU environments (e.g., RTX 4090 with 24GB VRAM).
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch
import gc
import os


def load_deepseek_r1_model(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    device_map: str = "auto",
    low_cpu_mem_usage: bool = True,
    max_memory: dict = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load DeepSeek-R1 model with 4-bit quantization for memory efficiency.
    
    Args:
        model_name: HuggingFace model identifier
        device_map: Device mapping strategy ("auto" recommended)
        low_cpu_mem_usage: Enable low CPU memory usage mode
        max_memory: Dict specifying max memory per device (e.g., {0: "20GiB", "cpu": "8GiB"})
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        RuntimeError: If model loading fails
    """
    print(f"Loading model: {model_name}")
    
    # Clear cache before loading
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Check for cgroup memory limit (common in Docker/containers)
    cgroup_limit_gb = None
    try:
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
            limit_bytes = int(f.read().strip())
            if limit_bytes < 2**63:  # Not unlimited
                cgroup_limit_gb = limit_bytes / (1024**3)
                print(f"⚠ Detected cgroup memory limit: {cgroup_limit_gb:.2f} GB")
    except:
        pass
    
    # Set default max_memory if not provided
    # Allow disabling ALL memory limits via environment variable for systems with lots of RAM
    disable_cpu_limit = os.getenv("DISABLE_CPU_MEMORY_LIMIT", "false").lower() == "true"
    disable_all_limits = os.getenv("DISABLE_ALL_MEMORY_LIMITS", "false").lower() == "true"
    
    if max_memory is None and not disable_all_limits:
        max_memory = {}
        if torch.cuda.is_available() and not disable_all_limits:
            # Limit GPU memory (leave some headroom)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            max_memory[0] = f"{int(gpu_memory_gb * 0.85)}GiB"  # More conservative
        
        # Limit CPU memory - respect cgroup limit if present
        if not disable_cpu_limit:
            if cgroup_limit_gb:
                # Use 80% of cgroup limit, leave headroom
                cpu_limit_gb = int(cgroup_limit_gb * 0.8)
                max_memory["cpu"] = f"{cpu_limit_gb}GiB"
                print(f"CPU memory limited to {cpu_limit_gb}GB (80% of cgroup limit)")
            else:
                max_memory["cpu"] = "4GiB"  # Default conservative limit
                print("CPU memory limited to 4GB (set DISABLE_CPU_MEMORY_LIMIT=true to disable)")
        else:
            print("CPU memory limit disabled (DISABLE_CPU_MEMORY_LIMIT=true)")
    
    if disable_all_limits:
        max_memory = None
        print("All memory limits disabled (DISABLE_ALL_MEMORY_LIMITS=true)")
        if cgroup_limit_gb:
            print(f"⚠ Warning: cgroup limit ({cgroup_limit_gb:.2f} GB) may still apply!")
    
    print("Configuring 4-bit quantization (NF4)...")
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with quantization
    print("Loading model with 4-bit quantization...")
    print(f"Max memory settings: {max_memory}")
    
    # Set environment variables for more aggressive memory management
    # Set before any CUDA operations
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Only create offload folder if we're using max_memory constraints
    offload_folder = None
    if max_memory:
        import tempfile
        offload_folder = tempfile.mkdtemp(prefix="model_offload_")
        print(f"Using offload folder: {offload_folder}")
    
    try:
        # Build load kwargs - keep it simple
        load_kwargs = {
            "quantization_config": quantization_config,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "dtype": torch.float16,
        }
        
        # Only add constraints if max_memory is specified
        if max_memory:
            load_kwargs["max_memory"] = max_memory
            if offload_folder:
                load_kwargs["offload_folder"] = offload_folder
        
        print("Attempting to load model...")
        print(f"Load kwargs: {list(load_kwargs.keys())}")
        
        # Try loading - this might fail with bitsandbytes bug, but we'll catch it
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        
        # Clear cache after loading
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Clean up offload folder (model is now loaded)
        if offload_folder:
            try:
                import shutil
                shutil.rmtree(offload_folder, ignore_errors=True)
            except:
                pass
        
        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        # Clear cache on error
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Clean up offload folder on error
        if offload_folder:
            try:
                import shutil
                shutil.rmtree(offload_folder, ignore_errors=True)
            except:
                pass
        
        raise RuntimeError(f"Failed to load model: {e}") from e

