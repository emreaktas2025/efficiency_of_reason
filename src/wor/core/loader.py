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
    
    # Set default max_memory if not provided (limit CPU to 4GB to prevent bad_alloc)
    if max_memory is None:
        max_memory = {}
        if torch.cuda.is_available():
            # Limit GPU memory (leave some headroom)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            max_memory[0] = f"{int(gpu_memory_gb * 0.85)}GiB"  # More conservative
        # Limit CPU memory aggressively to prevent std::bad_alloc
        max_memory["cpu"] = "4GiB"  # Reduced from 8GiB
    
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
    
    # Create temporary directory for offloading if needed
    import tempfile
    offload_folder = tempfile.mkdtemp(prefix="model_offload_")
    print(f"Using offload folder: {offload_folder}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
            max_memory=max_memory,
            offload_folder=offload_folder,  # Offload to disk if needed
            trust_remote_code=True,
            dtype=torch.float16,  # Fixed: use dtype instead of torch_dtype
            use_cache=False,  # Disable KV cache during loading to save memory
            use_safetensors=True,  # Use safetensors for more efficient loading
        )
        
        # Clear cache after loading
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Clean up offload folder (model is now loaded)
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
        try:
            import shutil
            shutil.rmtree(offload_folder, ignore_errors=True)
        except:
            pass
        
        raise RuntimeError(f"Failed to load model: {e}") from e

