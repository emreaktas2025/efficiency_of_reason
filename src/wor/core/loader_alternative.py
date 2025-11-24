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
    model_name: str = None,
    use_8bit: bool = False,
) -> tuple:
    """
    Alternative loader that tries 8-bit quantization or no quantization.
    
    Args:
        model_name: HuggingFace model identifier (defaults to 1.5B model for lower memory)
        use_8bit: If True, use 8-bit quantization instead of 4-bit
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Default to smaller model for constrained environments
    if model_name is None:
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    
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
        # For constrained containers, try loading WITHOUT quantization first
        # bitsandbytes needs too much CPU RAM during init
        # 1.5B model in FP16 only needs ~3GB GPU VRAM, which should fit
        if cgroup_limit_gb and cgroup_limit_gb < 48:
            print("⚠ Low memory container - trying WITHOUT quantization first...")
            print("  Loading directly to GPU (FP16) - 1.5B model needs ~3GB VRAM")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    dtype=torch.float16,
                    use_safetensors=True,
                )
                print("✓ Loaded without quantization (FP16 on GPU)")
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    print("⚠ GPU OOM - trying with 8-bit quantization...")
                    # Fallback to quantization if GPU doesn't have enough VRAM
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
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
        elif use_8bit:
            # For systems with more RAM, use quantization if requested
            print("Attempting 8-bit quantization...")
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
                use_safetensors=True,
            )
        else:
            print("Attempting to load without quantization (will use more memory)...")
            # For constrained containers, try without max_memory first
            if cgroup_limit_gb and cgroup_limit_gb < 48:
                print("⚠ Low memory container - trying without max_memory first...")
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        offload_folder=offload_folder,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        dtype=torch.float16,
                        use_safetensors=True,
                    )
                except RuntimeError as e:
                    if "bad_alloc" in str(e) or "memory" in str(e).lower():
                        print("⚠ Trying with max_memory as fallback...")
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
                    else:
                        raise
            else:
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

