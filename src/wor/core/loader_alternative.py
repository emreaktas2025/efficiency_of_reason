"""
Alternative model loader that tries different quantization strategies.

This is a fallback if the standard loader fails due to bitsandbytes issues.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
import os
import tempfile


def get_cgroup_limit_gb():
    """Get cgroup memory limit in GB, or None if unlimited."""
    try:
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
            limit_bytes = int(f.read().strip())
            if limit_bytes < 2**63:  # Not unlimited
                return limit_bytes / (1024**3)
    except:
        pass
    return None


def load_deepseek_r1_model_alternative(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    use_8bit: bool = False,
) -> tuple:
    """
    Alternative loader that tries 8-bit quantization or no quantization.
    
    Args:
        model_name: HuggingFace model identifier
        use_8bit: If True, use 8-bit quantization instead of 4-bit
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model (alternative method): {model_name}")
    
    # Clear cache
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Check cgroup limits
    cgroup_limit_gb = get_cgroup_limit_gb()
    if cgroup_limit_gb:
        print(f"⚠ Detected cgroup memory limit: {cgroup_limit_gb:.2f} GB")
    
    # Build max_memory constraints
    max_memory = {}
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        max_memory[0] = f"{int(gpu_memory_gb * 0.85)}GiB"
    
    # Set CPU limit based on cgroup
    if cgroup_limit_gb:
        if cgroup_limit_gb < 48:
            cpu_limit_gb = max(4, int(cgroup_limit_gb * 0.2))  # Very aggressive: 20%
            print(f"⚠ Container memory constrained - using {cpu_limit_gb}GB for CPU")
        else:
            cpu_limit_gb = int(cgroup_limit_gb * 0.5)
        max_memory["cpu"] = f"{cpu_limit_gb}GiB"
    else:
        max_memory["cpu"] = "8GiB"
    
    print(f"Max memory settings: {max_memory}")
    
    # Create offload folder
    offload_folder = tempfile.mkdtemp(prefix="model_offload_")
    print(f"Using offload folder: {offload_folder}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    try:
        # Try different loading strategies
        if use_8bit:
            print("Attempting 8-bit quantization with memory limits...")
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                max_memory=max_memory,
                offload_folder=offload_folder,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            print("Attempting to load without quantization (will use more memory)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                max_memory=max_memory,
                offload_folder=offload_folder,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )
        
        print("Model loaded successfully!")
        
        # Cleanup
        try:
            import shutil
            shutil.rmtree(offload_folder, ignore_errors=True)
        except:
            pass
        
        return model, tokenizer
        
    except Exception as e:
        # Cleanup
        try:
            import shutil
            shutil.rmtree(offload_folder, ignore_errors=True)
        except:
            pass
        
        if cgroup_limit_gb and cgroup_limit_gb < 48:
            print("\n" + "="*80)
            print("MEMORY ERROR - YOUR CONTAINER HAS INSUFFICIENT RAM")
            print("="*80)
            print(f"\nContainer limit: {cgroup_limit_gb:.1f}GB")
            print("Minimum required for this model: ~48GB")
            print("\nSOLUTION: Upgrade to a larger RunPod instance with 48GB+ RAM")
            print("="*80 + "\n")
        
        raise

