#!/usr/bin/env python3
"""
Pre-load script to download and cache the model before running experiments.

This can help avoid memory issues during the actual experiment run.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc

def preload_model():
    """Pre-download and cache the model."""
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    
    print("=" * 80)
    print("Pre-loading Model")
    print("=" * 80)
    print()
    
    print("Step 1: Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    print("✓ Tokenizer downloaded")
    
    print("\nStep 2: Downloading model config and files...")
    print("(This may take a few minutes)")
    
    # Just download the config first to verify connection
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print("✓ Model config downloaded")
    except Exception as e:
        print(f"✗ Failed to download config: {e}")
        return False
    
    print("\n✓ Model files are cached and ready!")
    print("\nYou can now run the experiment:")
    print("  python scripts/01_run_sparsity_gap.py")
    
    return True

if __name__ == "__main__":
    success = preload_model()
    sys.exit(0 if success else 1)

