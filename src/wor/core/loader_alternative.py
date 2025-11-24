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
    
    # Set CPU limit based on cgroup - VERY aggressive for constrained containers
    if cgroup_limit_gb:
        if cgroup_limit_gb < 48:
            # Even more aggressive: use only 3-4GB for CPU to force maximum offloading
            cpu_limit_gb = max(3, int(cgroup_limit_gb * 0.1))  # Only 10% for CPU!
            print(f"⚠ Container memory constrained - using {cpu_limit_gb}GB for CPU (10% of limit)")
            print(f"  This forces maximum disk offloading to avoid OOM")
        else:
            cpu_limit_gb = int(cgroup_limit_gb * 0.5)
        max_memory["cpu"] = f"{cpu_limit_gb}GiB"
    else:
        max_memory["cpu"] = "4GiB"  # Conservative default
    
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
            
            # First try with max_memory
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    max_memory=max_memory,
                    offload_folder=offload_folder,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
            except RuntimeError as e:
                if "bad_alloc" in str(e) or "memory" in str(e).lower():
                    print("⚠ max_memory approach failed, trying without max_memory (rely on device_map='auto')...")
                    # Fallback: let device_map handle it without explicit limits
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        offload_folder=offload_folder,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        use_safetensors=True,
                    )
                else:
                    raise
        else:
            print("Attempting to load without quantization (will use more memory)...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    max_memory=max_memory,
                    offload_folder=offload_folder,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    dtype=torch.float16,
                    use_safetensors=True,
                )
            except RuntimeError as e:
                if "bad_alloc" in str(e) or "memory" in str(e).lower():
                    print("⚠ max_memory approach failed, trying without max_memory...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        offload_folder=offload_folder,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        dtype=torch.float16,
                        use_safetensors=True,
                    )
                else:
                    raise
        
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

