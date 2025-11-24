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


def load_deepseek_r1_model(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    device_map: str = "auto",
    low_cpu_mem_usage: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load DeepSeek-R1 model with 4-bit quantization for memory efficiency.
    
    Args:
        model_name: HuggingFace model identifier
        device_map: Device mapping strategy ("auto" recommended)
        low_cpu_mem_usage: Enable low CPU memory usage mode
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        RuntimeError: If model loading fails
    """
    print(f"Loading model: {model_name}")
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
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

