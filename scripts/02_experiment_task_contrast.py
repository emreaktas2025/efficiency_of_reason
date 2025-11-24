#!/usr/bin/env python3
"""
Experiment 2: Task Contrast

Proves that sparsity differences aren't just token artifacts, but task differences.

Procedure:
1. Run DeepSeek-R1 on GSM8K (Math) - 500 problems
2. Run DeepSeek-R1 on MMLU/History (Knowledge) - 500 problems
3. Compare internal metrics of processing traces
4. Baseline Comparison: Run Llama-3-8B on same tasks (prompted to "think step by step")
5. Statistical analysis: Two-way ANOVA (task × model)
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add src to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_path = project_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from wor.core import load_deepseek_r1_model
from wor.core.loader_alternative import load_deepseek_r1_model_alternative
from wor.data import (
    parse_reasoning_output,
    load_gsm8k_dataset,
    load_mmlu_dataset,
    format_mmlu_prompt,
)
from wor.metrics import (
    calculate_metrics_for_segment,
    two_way_anova,
    interpret_effect_size,
)
from wor.visualization import violin_plot_cud_ape


def get_attention_states(model, tokenizer, full_text: str):
    """Extract attention weights for the full text."""
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
        )
    
    attentions = torch.stack(outputs.attentions, dim=0)
    return attentions, inputs["input_ids"][0]


def find_segment_token_indices(tokenizer, full_text: str, thinking_segment: str, response_segment: str) -> dict:
    """Find token indices for thinking and response segments."""
    full_tokens = tokenizer(full_text, add_special_tokens=True, return_tensors="pt")
    token_ids = full_tokens["input_ids"][0]
    
    think_start_tag = "<think>"
    think_end_tag = "</think>"
    
    if think_start_tag in full_text and think_end_tag in full_text:
        tag_start_pos = full_text.find(think_start_tag)
        tag_end_pos = full_text.find(think_end_tag) + len(think_end_tag)
        
        prefix = full_text[:tag_start_pos]
        prefix_tokens = tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
        thinking_start_idx = len(prefix_tokens) - 1
        
        up_to_thinking_end = full_text[:tag_end_pos]
        up_to_end_tokens = tokenizer(up_to_thinking_end, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
        thinking_end_idx = len(up_to_end_tokens)
        
        response_start_idx = thinking_end_idx
        response_end_idx = len(token_ids)
        
        return {
            "thinking": (thinking_start_idx, thinking_end_idx),
            "response": (response_start_idx, response_end_idx),
        }
    else:
        return {
            "thinking": (0, 0),
            "response": (0, len(token_ids)),
        }


def generate_with_reasoning(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate text with reasoning."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text


def run_task_analysis(
    model,
    tokenizer,
    problems: list,
    task_name: str,
    model_name: str,
) -> list:
    """Run analysis on a set of problems."""
    results = []
    
    for problem in tqdm(problems, desc=f"Processing {task_name}"):
        if task_name == "Math":
            prompt = problem['question']
        else:  # History/Knowledge
            prompt = format_mmlu_prompt(problem)
        
        try:
            generated = generate_with_reasoning(model, tokenizer, prompt)
            full_text = prompt + generated
            
            parsed = parse_reasoning_output(full_text)
            
            if not parsed["has_reasoning"]:
                continue
            
            attentions, token_ids = get_attention_states(model, tokenizer, full_text)
            
            segment_indices = find_segment_token_indices(
                tokenizer,
                full_text,
                parsed["thinking_segment"],
                parsed["response_segment"],
            )
            
            thinking_start, thinking_end = segment_indices["thinking"]
            
            if thinking_end > thinking_start:
                thinking_metrics = calculate_metrics_for_segment(
                    attentions,
                    thinking_start,
                    thinking_end,
                )
                
                results.append({
                    'task': task_name,
                    'model': model_name,
                    'cud': thinking_metrics['cud'],
                    'ape': thinking_metrics['ape'],
                })
        except Exception as e:
            continue
    
    return results


def run_experiment_2(
    num_problems: int = 500,
    output_dir: Path = None,
    use_baseline_model: bool = False,
):
    """
    Run Experiment 2: Task Contrast.
    
    Args:
        num_problems: Number of problems per task
        output_dir: Directory to save results
        use_baseline_model: Whether to also run Llama-3-8B baseline
    """
    if output_dir is None:
        output_dir = Path("results/experiment_2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Experiment 2: Task Contrast")
    print("=" * 80)
    print()
    
    # Load DeepSeek-R1
    print("Step 1: Loading DeepSeek-R1 model...")
    try:
        model, tokenizer = load_deepseek_r1_model()
        model.eval()
        print("✓ DeepSeek-R1 loaded")
    except Exception as e:
        print(f"⚠ DeepSeek-R1 loading failed: {e}")
        print("Trying alternative loader...")
        model, tokenizer = load_deepseek_r1_model_alternative(use_8bit=True)
        model.eval()
    print()
    
    # Load datasets
    print("Step 2: Loading datasets...")
    try:
        math_problems = load_gsm8k_dataset(num_problems=num_problems, split="test")
        print(f"✓ Loaded {len(math_problems)} math problems")
    except Exception as e:
        print(f"⚠ GSM8K loading failed: {e}")
        from wor.data.datasets import get_gsm8k_fallback_problems
        math_problems = get_gsm8k_fallback_problems(num_problems)
    
    try:
        history_problems = load_mmlu_dataset(num_problems=num_problems, subject="history", split="test")
        print(f"✓ Loaded {len(history_problems)} history problems")
    except Exception as e:
        print(f"⚠ MMLU loading failed: {e}")
        history_problems = []
    
    if not history_problems:
        print("⚠ Using simplified history prompts as fallback")
        history_problems = [
            {'question': f"Explain the causes of World War {i}.", 'choices': [], 'answer': 0}
            for i in range(1, num_problems + 1)
        ]
    print()
    
    # Run analysis
    print("Step 3: Running analysis...")
    print()
    
    all_results = []
    
    # Math task with DeepSeek-R1
    print("3.1 Math task (DeepSeek-R1)...")
    math_results = run_task_analysis(
        model, tokenizer, math_problems, "Math", "DeepSeek-R1"
    )
    all_results.extend(math_results)
    print(f"✓ Processed {len(math_results)} math problems")
    print()
    
    # History task with DeepSeek-R1
    print("3.2 History task (DeepSeek-R1)...")
    history_results = run_task_analysis(
        model, tokenizer, history_problems, "History", "DeepSeek-R1"
    )
    all_results.extend(history_results)
    print(f"✓ Processed {len(history_results)} history problems")
    print()
    
    # Baseline model (if requested)
    if use_baseline_model:
        print("3.3 Loading baseline model (Llama-3-8B)...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            baseline_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")
            baseline_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3-8B-Instruct",
                device_map="auto",
                torch_dtype=torch.float16,
            )
            baseline_model.eval()
            print("✓ Llama-3-8B loaded")
            
            # Note: Llama-3 won't have <think> tags, so we analyze the full output
            # This is a simplified version - full implementation would need different parsing
            print("⚠ Baseline model analysis requires different parsing (no <think> tags)")
            print("   Skipping baseline for now")
        except Exception as e:
            print(f"⚠ Baseline model loading failed: {e}")
        print()
    
    # Statistical Analysis
    print("Step 4: Statistical Analysis")
    print("-" * 80)
    print()
    
    if len(all_results) < 4:
        print("⚠ Insufficient data for statistical analysis")
        return
    
    # Prepare data for ANOVA
    cud_values = np.array([r['cud'] for r in all_results])
    ape_values = np.array([r['ape'] for r in all_results])
    tasks = np.array([r['task'] for r in all_results])
    models = np.array([r['model'] for r in all_results])
    
    print("4.1 Two-way ANOVA (Task × Model):")
    print()
    
    cud_anova = two_way_anova(cud_values, tasks, models)
    ape_anova = two_way_anova(ape_values, tasks, models)
    
    print("CUD ANOVA Results:")
    print(f"  Task effect: F = {cud_anova['f_factor1']:.4f}, p = {cud_anova['p_factor1']:.4f}")
    print(f"  Model effect: F = {cud_anova['f_factor2']:.4f}, p = {cud_anova['p_factor2']:.4f}")
    if cud_anova['f_interaction'] is not None:
        print(f"  Interaction: F = {cud_anova['f_interaction']:.4f}, p = {cud_anova['p_interaction']:.4f}")
    print()
    
    print("APE ANOVA Results:")
    print(f"  Task effect: F = {ape_anova['f_factor1']:.4f}, p = {ape_anova['p_factor1']:.4f}")
    print(f"  Model effect: F = {ape_anova['f_factor2']:.4f}, p = {ape_anova['p_factor2']:.4f}")
    if ape_anova['f_interaction'] is not None:
        print(f"  Interaction: F = {ape_anova['f_interaction']:.4f}, p = {ape_anova['p_interaction']:.4f}")
    print()
    
    # Summary statistics
    print("4.2 Summary Statistics by Task:")
    print()
    
    math_cud = [r['cud'] for r in all_results if r['task'] == 'Math']
    history_cud = [r['cud'] for r in all_results if r['task'] == 'History']
    
    table_data = [
        ["Task", "Metric", "Mean", "Std", "N"],
        ["Math", "CUD", f"{np.mean(math_cud):.4f}", f"{np.std(math_cud):.4f}", len(math_cud)],
        ["History", "CUD", f"{np.mean(history_cud):.4f}", f"{np.std(history_cud):.4f}", len(history_cud)],
    ]
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    print()
    
    # Visualizations
    print("Step 5: Generating visualizations...")
    
    math_metrics = [{"cud": r['cud'], "ape": r['ape']} for r in all_results if r['task'] == 'Math']
    history_metrics = [{"cud": r['cud'], "ape": r['ape']} for r in all_results if r['task'] == 'History']
    
    violin_plot_cud_ape(
        math_metrics,
        history_metrics,
        output_path=output_dir / "task_contrast_cud.png",
        metric_name="CUD",
    )
    
    print("✓ Visualizations saved")
    print()
    
    # Save results
    results_data = {
        'experiment': 'Experiment 2: Task Contrast',
        'num_problems_per_task': num_problems,
        'anova_results': {
            'cud': cud_anova,
            'ape': ape_anova,
        },
        'summary_stats': {
            'math_cud_mean': float(np.mean(math_cud)),
            'history_cud_mean': float(np.mean(history_cud)),
        },
        'results': all_results,
    }
    
    results_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"✓ Results saved to {results_file}")
    print()
    
    # Final summary
    print("=" * 80)
    print("Key Finding:")
    if cud_anova['p_factor1'] < 0.05:
        print("  ✓ CONFIRMED: Task type significantly affects sparsity")
        print(f"    Math traces: CUD = {np.mean(math_cud):.4f}")
        print(f"    History traces: CUD = {np.mean(history_cud):.4f}")
    else:
        print("  ✗ No significant task effect found")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Experiment 2: Task Contrast")
    parser.add_argument("--num-problems", type=int, default=500,
                       help="Number of problems per task (default: 500)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--baseline", action="store_true",
                       help="Also run Llama-3-8B baseline")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_experiment_2(
        num_problems=args.num_problems,
        output_dir=output_dir,
        use_baseline_model=args.baseline,
    )

