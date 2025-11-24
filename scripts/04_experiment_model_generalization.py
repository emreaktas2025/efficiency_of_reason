#!/usr/bin/env python3
"""
Experiment 4: Model Generalization

Tests if sparsity is a general property of reasoning, or DeepSeek-R1 specific.

Procedure:
1. Run same experiments on DeepSeek-R1-1.5B (if available) or Qwen-2.5-Math
2. Compare sparsity patterns across model sizes
3. Meta-analysis: Are effect sizes consistent?
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

from wor.core.loader_alternative import load_deepseek_r1_model_alternative
from wor.data import parse_reasoning_output, load_gsm8k_dataset
from wor.metrics import calculate_metrics_for_segment, paired_t_test, interpret_effect_size


def run_model_analysis(
    model_name: str,
    num_problems: int = 100,
) -> dict:
    """
    Run sparsity analysis on a specific model.
    
    Args:
        model_name: HuggingFace model identifier
        num_problems: Number of problems to analyze
        
    Returns:
        Dictionary with results
    """
    print(f"\nAnalyzing model: {model_name}")
    print("-" * 80)
    
    # Load model (simplified - assumes it can be loaded similarly)
    # For different models, you'd need model-specific loading
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Try 4-bit quantization
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        except:
            # Fallback to 8-bit
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        
        model.eval()
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None
    
    # Load problems
    try:
        problems = load_gsm8k_dataset(num_problems=num_problems, split="test")
    except:
        from wor.data.datasets import get_gsm8k_fallback_problems
        problems = get_gsm8k_fallback_problems(num_problems)
    
    # Run analysis (simplified - reuse logic from Experiment 1)
    thinking_cud = []
    response_cud = []
    
    for problem in tqdm(problems[:min(50, len(problems))], desc="Processing"):
        prompt = problem['question']
        
        try:
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
                # For models without <think> tags, analyze full output as "thinking"
                # This is a simplification
                continue
            
            # Get attention
            inputs_full = tokenizer(full_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs_full = model(**inputs_full, output_attentions=True)
            
            attentions = torch.stack(outputs_full.attentions, dim=0)
            
            # Find segments (simplified)
            think_start = full_text.find("<think>")
            think_end = full_text.find("</think>")
            if think_start == -1 or think_end == -1:
                continue
            
            prefix = full_text[:think_start]
            prefix_tokens = tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
            thinking_start_idx = len(prefix_tokens) - 1
            
            up_to_end = full_text[:think_end + len("</think>")]
            up_to_end_tokens = tokenizer(up_to_end, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
            thinking_end_idx = len(up_to_end_tokens)
            response_start_idx = thinking_end_idx
            response_end_idx = len(tokenizer(full_text, add_special_tokens=True, return_tensors="pt")["input_ids"][0])
            
            # Calculate metrics
            if thinking_end_idx > thinking_start_idx:
                thinking_metrics = calculate_metrics_for_segment(
                    attentions,
                    thinking_start_idx,
                    thinking_end_idx,
                )
                thinking_cud.append(thinking_metrics['cud'])
            
            if response_end_idx > response_start_idx:
                response_metrics = calculate_metrics_for_segment(
                    attentions,
                    response_start_idx,
                    response_end_idx,
                )
                response_cud.append(response_metrics['cud'])
        
        except Exception as e:
            continue
    
    if len(thinking_cud) < 2 or len(response_cud) < 2:
        print("⚠ Insufficient data")
        return None
    
    # Statistical test
    test_result = paired_t_test(np.array(thinking_cud), np.array(response_cud))
    
    return {
        'model_name': model_name,
        'n': len(thinking_cud),
        'thinking_cud_mean': float(np.mean(thinking_cud)),
        'response_cud_mean': float(np.mean(response_cud)),
        'mean_diff': test_result['mean_diff'],
        'cohens_d': test_result['cohens_d'],
        'p_value': test_result['p_value'],
        'effect_size_label': interpret_effect_size(test_result['cohens_d']),
    }


def run_experiment_4(
    output_dir: Path = None,
):
    """
    Run Experiment 4: Model Generalization.
    
    Args:
        output_dir: Directory to save results
    """
    if output_dir is None:
        output_dir = Path("results/experiment_4")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Experiment 4: Model Generalization")
    print("=" * 80)
    print()
    
    models_to_test = [
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    ]
    
    all_results = []
    
    for model_name in models_to_test:
        result = run_model_analysis(model_name, num_problems=100)
        if result:
            all_results.append(result)
    
    if not all_results:
        print("⚠ No models successfully analyzed")
        return
    
    # Meta-analysis
    print("\n" + "=" * 80)
    print("Meta-Analysis: Effect Size Consistency")
    print("=" * 80)
    print()
    
    table_data = [
        ["Model", "N", "Thinking CUD", "Response CUD", "Cohen's d", "Effect Size", "p-value"],
    ]
    
    for result in all_results:
        table_data.append([
            result['model_name'].split('/')[-1],
            result['n'],
            f"{result['thinking_cud_mean']:.4f}",
            f"{result['response_cud_mean']:.4f}",
            f"{result['cohens_d']:.4f}",
            result['effect_size_label'],
            f"{result['p_value']:.4f}",
        ])
    
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    print()
    
    # Check consistency
    effect_sizes = [r['cohens_d'] for r in all_results]
    mean_effect = np.mean(effect_sizes)
    std_effect = np.std(effect_sizes)
    
    print("Effect Size Consistency:")
    print(f"  Mean Cohen's d: {mean_effect:.4f}")
    print(f"  Std Cohen's d: {std_effect:.4f}")
    
    if std_effect < 0.2:
        print("  ✓ Effect sizes are consistent across models")
    else:
        print("  ⚠ Effect sizes vary across models")
    print()
    
    # Save results
    results = {
        'experiment': 'Experiment 4: Model Generalization',
        'models_tested': [r['model_name'] for r in all_results],
        'results': all_results,
        'meta_analysis': {
            'mean_effect_size': float(mean_effect),
            'std_effect_size': float(std_effect),
        },
    }
    
    results_file = output_dir / "generalization_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {results_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Experiment 4: Model Generalization")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_experiment_4(output_dir=output_dir)

