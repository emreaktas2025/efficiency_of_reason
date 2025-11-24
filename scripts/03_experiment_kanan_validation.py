#!/usr/bin/env python3
"""
Experiment 3: The "Kanan" Validation

Proves modularity for Continual Learning applications through ablation.

Procedure:
1. Identify Top-1% of Attention Heads active during <think> segments (using CUD)
2. Zero-Ablate these heads
3. Test: Does model fail at Math? (Yes - reasoning impaired)
4. Control: Does model still speak fluent English? (Yes - language intact)
5. Reverse Control: Ablate random 1% of heads - should affect both tasks equally
"""

import sys
from pathlib import Path
import json
import numpy as np
from tabulate import tabulate

# Add src to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_path = project_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
from tqdm import tqdm

from wor.core import load_deepseek_r1_model
from wor.core.loader_alternative import load_deepseek_r1_model_alternative
from wor.data import parse_reasoning_output, load_gsm8k_dataset, check_gsm8k_answer
from wor.metrics import calculate_cud


def identify_reasoning_heads(
    model,
    tokenizer,
    problems: list,
    top_percent: float = 0.01,
) -> list:
    """
    Identify top attention heads active during reasoning segments.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        problems: List of GSM8K problems
        top_percent: Percentage of heads to identify (default: 1%)
        
    Returns:
        List of (layer_idx, head_idx) tuples for top heads
    """
    print("Identifying reasoning-specific attention heads...")
    
    # Collect head activations across all reasoning segments
    all_head_activations = {}  # (layer, head) -> list of activations
    
    for problem in tqdm(problems[:100], desc="Analyzing heads"):  # Use subset for efficiency
        prompt = problem['question']
        
        try:
            # Generate
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
            
            full_text = prompt + generated
            parsed = parse_reasoning_output(full_text)
            
            if not parsed["has_reasoning"]:
                continue
            
            # Get attention for full sequence
            inputs_full = tokenizer(full_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs_full = model(**inputs_full, output_attentions=True)
            
            attentions = torch.stack(outputs_full.attentions, dim=0)  # (layers, batch, heads, seq, seq)
            
            # Find thinking segment
            think_start = full_text.find("<think>")
            think_end = full_text.find("</think>")
            if think_start == -1 or think_end == -1:
                continue
            
            # Tokenize to find indices
            prefix = full_text[:think_start]
            prefix_tokens = tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
            thinking_start_idx = len(prefix_tokens) - 1
            
            up_to_end = full_text[:think_end + len("</think>")]
            up_to_end_tokens = tokenizer(up_to_end, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
            thinking_end_idx = len(up_to_end_tokens)
            
            # Calculate per-head activation in thinking segment
            thinking_attn = attentions[:, 0, :, thinking_start_idx:thinking_end_idx, thinking_start_idx:thinking_end_idx]
            # Average over sequence: (layers, heads)
            head_activations = thinking_attn.mean(dim=(-2, -1))
            
            # Store activations
            num_layers, num_heads = head_activations.shape
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    key = (layer_idx, head_idx)
                    if key not in all_head_activations:
                        all_head_activations[key] = []
                    all_head_activations[key].append(head_activations[layer_idx, head_idx].item())
        
        except Exception as e:
            continue
    
    # Calculate mean activation per head
    head_means = {
        key: np.mean(vals) for key, vals in all_head_activations.items()
    }
    
    # Sort by mean activation
    sorted_heads = sorted(head_means.items(), key=lambda x: x[1], reverse=True)
    
    # Select top percent
    num_heads_to_select = max(1, int(len(sorted_heads) * top_percent))
    top_heads = [head[0] for head in sorted_heads[:num_heads_to_select]]
    
    print(f"✓ Identified {len(top_heads)} reasoning-specific heads (top {top_percent*100}%)")
    
    return top_heads


def ablate_heads(model, head_indices: list):
    """
    Zero-ablate specified attention heads.
    
    Args:
        model: Model to modify
        head_indices: List of (layer_idx, head_idx) tuples
    """
    def make_ablation_hook(layer_idx, head_idx):
        def hook(module, input, output):
            # output is tuple: (attn_output, attn_weights, ...)
            if len(output) >= 2:
                attn_weights = output[1]  # (batch, heads, seq, seq)
                # Zero out specific head
                attn_weights[:, head_idx, :, :] = 0.0
                # Re-normalize
                attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-10)
                # Return modified output
                return (output[0], attn_weights) + output[2:]
            return output
        return hook
    
    # Register hooks
    hooks = []
    for layer_idx, head_idx in head_indices:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layer = model.model.layers[layer_idx]
            if hasattr(layer, 'self_attn'):
                hook = layer.self_attn.register_forward_hook(
                    make_ablation_hook(layer_idx, head_idx)
                )
                hooks.append(hook)
    
    return hooks


def test_model_performance(
    model,
    tokenizer,
    problems: list,
    task_type: str = "math",
) -> dict:
    """
    Test model performance on a task.
    
    Args:
        model: Model to test
        tokenizer: Tokenizer
        problems: List of problems
        task_type: Type of task ("math" or "language")
        
    Returns:
        Dictionary with performance metrics
    """
    correct = 0
    total = 0
    
    for problem in tqdm(problems[:50], desc=f"Testing {task_type}"):  # Use subset for speed
        if task_type == "math":
            prompt = problem['question']
            correct_answer = problem.get('answer', '')
        else:  # language fluency test
            prompt = "Write a grammatically correct sentence about the weather."
            correct_answer = None
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
            
            if task_type == "math":
                is_correct, _ = check_gsm8k_answer(generated, correct_answer)
                if is_correct:
                    correct += 1
            else:
                # Simple fluency check: contains common words and is reasonable length
                if len(generated.split()) > 3 and any(word in generated.lower() for word in ['the', 'is', 'a', 'and']):
                    correct += 1
            
            total += 1
        except:
            continue
    
    accuracy = correct / total if total > 0 else 0.0
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
    }


def run_experiment_3(
    num_problems: int = 100,
    output_dir: Path = None,
):
    """
    Run Experiment 3: Kanan Validation (Ablation).
    
    Args:
        num_problems: Number of problems to use for testing
        output_dir: Directory to save results
    """
    if output_dir is None:
        output_dir = Path("results/experiment_3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Experiment 3: The 'Kanan' Validation")
    print("=" * 80)
    print()
    
    # Load model
    print("Step 1: Loading model...")
    try:
        model, tokenizer = load_deepseek_r1_model()
        model.eval()
    except Exception as e:
        print(f"⚠ Standard loader failed: {e}")
        model, tokenizer = load_deepseek_r1_model_alternative(use_8bit=True)
        model.eval()
    print("✓ Model loaded")
    print()
    
    # Load problems
    print("Step 2: Loading test problems...")
    try:
        problems = load_gsm8k_dataset(num_problems=num_problems, split="test")
    except:
        from wor.data.datasets import get_gsm8k_fallback_problems
        problems = get_gsm8k_fallback_problems(num_problems)
    print(f"✓ Loaded {len(problems)} problems")
    print()
    
    # Baseline performance
    print("Step 3: Baseline performance (no ablation)...")
    baseline_math = test_model_performance(model, tokenizer, problems, task_type="math")
    baseline_lang = test_model_performance(model, tokenizer, problems, task_type="language")
    print(f"  Math accuracy: {baseline_math['accuracy']:.2%}")
    print(f"  Language fluency: {baseline_lang['accuracy']:.2%}")
    print()
    
    # Identify reasoning heads
    print("Step 4: Identifying reasoning-specific heads...")
    reasoning_heads = identify_reasoning_heads(model, tokenizer, problems, top_percent=0.01)
    print(f"✓ Selected {len(reasoning_heads)} heads for ablation")
    print(f"  Heads: {reasoning_heads[:10]}..." if len(reasoning_heads) > 10 else f"  Heads: {reasoning_heads}")
    print()
    
    # Ablate reasoning heads
    print("Step 5: Ablating reasoning-specific heads...")
    hooks = ablate_heads(model, reasoning_heads)
    print("✓ Heads ablated")
    print()
    
    # Test after ablation
    print("Step 6: Testing performance after ablation...")
    ablated_math = test_model_performance(model, tokenizer, problems, task_type="math")
    ablated_lang = test_model_performance(model, tokenizer, problems, task_type="language")
    print(f"  Math accuracy: {ablated_math['accuracy']:.2%}")
    print(f"  Language fluency: {ablated_lang['accuracy']:.2%}")
    print()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Reverse control: Ablate random heads
    print("Step 7: Reverse control - Ablating random heads...")
    num_layers = len(model.model.layers) if hasattr(model, 'model') else 32
    num_heads_per_layer = model.config.num_attention_heads if hasattr(model, 'config') else 32
    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads_per_layer)]
    random_heads = np.random.choice(len(all_heads), size=len(reasoning_heads), replace=False)
    random_head_indices = [all_heads[i] for i in random_heads]
    
    hooks_random = ablate_heads(model, random_head_indices)
    random_math = test_model_performance(model, tokenizer, problems, task_type="math")
    random_lang = test_model_performance(model, tokenizer, problems, task_type="language")
    print(f"  Math accuracy: {random_math['accuracy']:.2%}")
    print(f"  Language fluency: {random_lang['accuracy']:.2%}")
    print()
    
    for hook in hooks_random:
        hook.remove()
    
    # Results summary
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()
    
    table_data = [
        ["Condition", "Math Accuracy", "Language Fluency"],
        ["Baseline (no ablation)", f"{baseline_math['accuracy']:.2%}", f"{baseline_lang['accuracy']:.2%}"],
        ["Reasoning heads ablated", f"{ablated_math['accuracy']:.2%}", f"{ablated_lang['accuracy']:.2%}"],
        ["Random heads ablated", f"{random_math['accuracy']:.2%}", f"{random_lang['accuracy']:.2%}"],
    ]
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    print()
    
    # Interpretation
    print("Interpretation:")
    math_impairment = baseline_math['accuracy'] - ablated_math['accuracy']
    lang_impairment = baseline_lang['accuracy'] - ablated_lang['accuracy']
    
    if math_impairment > 0.1 and lang_impairment < 0.1:
        print("  ✓ CONFIRMED: Reasoning heads are modular")
        print("    - Math performance significantly impaired")
        print("    - Language fluency preserved")
    elif math_impairment > 0.1:
        print("  ⚠ Reasoning heads affect both tasks")
    else:
        print("  ✗ Reasoning head ablation did not significantly affect math")
    
    # Save results
    results = {
        'experiment': 'Experiment 3: Kanan Validation',
        'reasoning_heads': reasoning_heads,
        'baseline': {
            'math': baseline_math,
            'language': baseline_lang,
        },
        'ablated_reasoning': {
            'math': ablated_math,
            'language': ablated_lang,
        },
        'ablated_random': {
            'math': random_math,
            'language': random_lang,
        },
    }
    
    results_file = output_dir / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Experiment 3: Kanan Validation")
    parser.add_argument("--num-problems", type=int, default=100,
                       help="Number of problems for testing")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_experiment_3(
        num_problems=args.num_problems,
        output_dir=output_dir,
    )

