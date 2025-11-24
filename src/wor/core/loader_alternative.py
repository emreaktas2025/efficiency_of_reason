"""
Alternative model loader that tries different quantization strategies.

This is a fallback if the standard loader fails due to bitsandbytes issues.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc


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
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Try different loading strategies
    if use_8bit:
        print("Attempting 8-bit quantization...")
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.float16,
        )
    else:
        print("Attempting to load without quantization (will use more memory)...")
        # Try loading without device_map first, then move to GPU manually
        print("Loading model to CPU first...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,  # Load to CPU first
            trust_remote_code=True,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            print("Moving model to GPU...")
            model = model.to("cuda")
    
    print("Model loaded successfully!")
    return model, tokenizer

