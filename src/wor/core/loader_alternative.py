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
    # Default to 8B model (better for research, works with 24GB+ VRAM)
    # Use 1.5B only if explicitly set via DEEPSEEK_MODEL env var
    if model_name is None:
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    
    print(f"Loading model (alternative method): {model_name}")
    
    # Clear cache
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Check cgroup limits
    cgroup_limit_gb = get_cgroup_limit_gb()
    if cgroup_limit_gb:
        print(f"⚠ Detected cgroup memory limit: {cgroup_limit_gb:.2f} GB")
    
    # For constrained containers, load directly to GPU WITHOUT any CPU memory limits
    # The model loading process needs temporary CPU RAM to process weights before GPU transfer
    # Setting max_memory limits causes std::bad_alloc even for small models
    use_direct_gpu_load = cgroup_limit_gb and cgroup_limit_gb < 48
    
    if use_direct_gpu_load:
        # NO memory constraints - let the system use what it needs during loading
        max_memory = None
        offload_folder = None
        print("⚠ Low memory container - loading directly to GPU")
        print("  NO CPU memory limits (model needs ~6GB temp RAM during load)")
        print("  Loading straight to GPU (FP16) - will use ~3GB VRAM")
    else:
        # Build max_memory constraints only for systems with more RAM
        max_memory = {}
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            max_memory[0] = f"{int(gpu_memory_gb * 0.85)}GiB"
        
        if cgroup_limit_gb:
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
        # For constrained containers, try loading WITHOUT quantization first
        # bitsandbytes needs too much CPU RAM during init
        # 1.5B model in FP16 only needs ~3GB GPU VRAM, which should fit
        if use_direct_gpu_load:
            print("Loading model directly to GPU (no CPU memory limits)...")
            try:
                # Load directly to GPU with NO max_memory constraints
                # The loading process needs temporary CPU RAM (~6GB) to process weights
                # Setting max_memory causes std::bad_alloc even for small models
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="cuda:0",  # Force immediate GPU load
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                )
                print("✓ Model loaded successfully (FP16 on GPU)")
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    print("⚠ GPU OOM - trying with device_map='auto' and offloading...")
                    # Fallback: use auto device_map with offloading
                    import tempfile
                    offload_folder = tempfile.mkdtemp(prefix="model_offload_")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        offload_folder=offload_folder,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                    )
                else:
                    raise
        elif use_8bit and not use_direct_gpu_load:
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

