#!/usr/bin/env python3
"""
Quick test script to verify model loading with different strategies.

This is useful for debugging memory issues before running the full experiment.
"""

import sys
import os
from pathlib import Path

# Add src to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import gc


def print_memory_info():
    """Print current memory usage."""
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
        print(f"GPU Memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")

    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"CPU Memory: {mem_info.rss / 1e9:.2f} GB used by process")
    except ImportError:
        pass


def test_4bit_loading():
    """Test 4-bit quantization (default)."""
    print("\n" + "="*80)
    print("Testing 4-bit Quantization (Default Loader)")
    print("="*80 + "\n")

    from wor.core import load_deepseek_r1_model

    try:
        model, tokenizer = load_deepseek_r1_model()
        print("\n✅ Model loaded successfully with 4-bit quantization!")
        print_memory_info()

        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return True
    except Exception as e:
        print(f"\n❌ Failed to load with 4-bit quantization: {e}")
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return False


def test_8bit_loading():
    """Test 8-bit quantization (alternative)."""
    print("\n" + "="*80)
    print("Testing 8-bit Quantization (Alternative Loader)")
    print("="*80 + "\n")

    from wor.core.loader_alternative import load_deepseek_r1_model_alternative

    try:
        model, tokenizer = load_deepseek_r1_model_alternative(use_8bit=True)
        print("\n✅ Model loaded successfully with 8-bit quantization!")
        print_memory_info()

        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return True
    except Exception as e:
        print(f"\n❌ Failed to load with 8-bit quantization: {e}")
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return False


def test_no_quant_loading():
    """Test loading without quantization (FP16)."""
    print("\n" + "="*80)
    print("Testing FP16 (No Quantization)")
    print("="*80 + "\n")

    from wor.core.loader_alternative import load_deepseek_r1_model_alternative

    try:
        model, tokenizer = load_deepseek_r1_model_alternative(use_8bit=False)
        print("\n✅ Model loaded successfully in FP16!")
        print_memory_info()

        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return True
    except Exception as e:
        print(f"\n❌ Failed to load in FP16: {e}")
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return False


def main():
    """Run loading tests based on environment variables."""
    print("Model Loading Test Script")
    print("=" * 80)

    # Check environment
    use_alternative = os.getenv("USE_ALTERNATIVE_LOADER", "false").lower() == "true"
    use_8bit = os.getenv("USE_8BIT_QUANTIZATION", "false").lower() == "true"

    print("\nEnvironment:")
    print(f"  USE_ALTERNATIVE_LOADER: {use_alternative}")
    print(f"  USE_8BIT_QUANTIZATION: {use_8bit}")
    print(f"  DISABLE_CPU_MEMORY_LIMIT: {os.getenv('DISABLE_CPU_MEMORY_LIMIT', 'false')}")
    print(f"  DISABLE_ALL_MEMORY_LIMITS: {os.getenv('DISABLE_ALL_MEMORY_LIMITS', 'false')}")

    # Check system
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"\nGPU: {gpu_props.name}")
        print(f"  VRAM: {gpu_props.total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠️  WARNING: No CUDA GPU detected!")

    # Check cgroup limit
    try:
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
            limit_bytes = int(f.read().strip())
            if limit_bytes < 2**63:
                limit_gb = limit_bytes / (1024**3)
                print(f"\nContainer RAM Limit: {limit_gb:.2f} GB")
    except:
        print("\nNo cgroup limit detected (not in container)")

    # Determine which test to run
    success = False

    if use_alternative:
        if use_8bit:
            success = test_8bit_loading()
        else:
            success = test_no_quant_loading()
    else:
        # Try 4-bit first
        success = test_4bit_loading()

        if not success:
            print("\n" + "="*80)
            print("4-bit loading failed. Trying 8-bit as fallback...")
            print("="*80)

            # Clear memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            success = test_8bit_loading()

            if success:
                print("\n" + "="*80)
                print("RECOMMENDATION: Use 8-bit quantization for this system")
                print("="*80)
                print("\nTo use 8-bit by default, run:")
                print("  export USE_ALTERNATIVE_LOADER=true")
                print("  export USE_8BIT_QUANTIZATION=true")
                print("  python scripts/01_run_sparsity_gap.py")

    # Print final result
    print("\n" + "="*80)
    if success:
        print("✅ SUCCESS: Model loading works!")
        print("="*80)
        print("\nYou can now run the experiments:")
        print("  python scripts/01_run_sparsity_gap.py")
    else:
        print("❌ FAILURE: Could not load model with any strategy")
        print("="*80)
        print("\nPlease check TROUBLESHOOTING.md for solutions.")
        print("\nQuick fixes to try:")
        print("  1. Upgrade to a RunPod instance with 48GB+ RAM")
        print("  2. Try: export DISABLE_ALL_MEMORY_LIMITS=true")
        print("  3. Check available memory: free -h")
    print("="*80 + "\n")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

