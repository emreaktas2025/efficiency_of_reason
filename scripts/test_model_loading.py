#!/usr/bin/env python3
"""Test script to diagnose model loading issues."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import gc

def test_model_loading():
    """Test different model loading strategies."""
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    
    print("=" * 80)
    print("Model Loading Diagnostic Test")
    print("=" * 80)
    print()
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print()
    
    # Test 1: Load config
    print("Test 1: Loading model config...")
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ Config loaded: {config.model_type}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Num layers: {config.num_hidden_layers}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False
    print()
    
    # Test 2: Load tokenizer
    print("Test 2: Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return False
    print()
    
    # Test 3: Try loading model with minimal options
    print("Test 3: Loading model (minimal options, CPU only)...")
    gc.collect()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,  # CPU only
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("✓ Model loaded to CPU successfully!")
        print(f"  Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
        del model
        gc.collect()
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    print("=" * 80)
    print("All tests passed! Model can be loaded.")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)

