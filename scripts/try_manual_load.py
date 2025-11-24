#!/usr/bin/env python3
"""Try manually loading model weights to isolate the issue."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import snapshot_download
import os

def try_manual_load():
    """Try downloading and loading model files manually."""
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    
    print("=" * 80)
    print("Manual Model Loading Test")
    print("=" * 80)
    print()
    
    # Clear cache first
    cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"Cache directory: {cache_dir}")
    
    # Load config
    print("1. Loading config...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print(f"   ✓ Config: {config.model_type}, {config.num_hidden_layers} layers")
    
    # Try downloading model files
    print("\n2. Downloading model files...")
    try:
        model_path = snapshot_download(
            repo_id=model_name,
            allow_patterns=["*.safetensors", "*.bin", "*.json"],
            cache_dir=None,  # Use default
        )
        print(f"   ✓ Model files downloaded to: {model_path}")
    except Exception as e:
        print(f"   ✗ Failed to download: {e}")
        return False
    
    # Try loading just one weight file to see if it's a file issue
    print("\n3. Checking weight files...")
    import glob
    weight_files = glob.glob(f"{model_path}/model*.safetensors") + glob.glob(f"{model_path}/pytorch_model*.bin")
    print(f"   Found {len(weight_files)} weight files")
    
    if weight_files:
        print(f"   First file: {weight_files[0]}")
        file_size = os.path.getsize(weight_files[0]) / (1024**3)
        print(f"   Size: {file_size:.2f} GB")
    
    print("\n4. Trying to load with explicit local path...")
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,  # Use local path
            local_files_only=True,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=False,  # Try without this
        )
        print("   ✓ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {type(e).__name__}: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = try_manual_load()
    sys.exit(0 if success else 1)

