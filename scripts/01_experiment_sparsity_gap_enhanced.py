#!/usr/bin/env python3
"""
Experiment 1: The Sparsity Gap (Enhanced)

Enhanced version with statistical analysis, controls, and increased sample size
as specified in the enhanced research plan.

Procedure:
1. Load DeepSeek-R1 model with 4-bit quantization
2. Run inference on 500-1000 GSM8K problems
3. Extract internal attention states
4. Parse outputs into thinking vs response segments
5. Calculate CUD, APE, and AE metrics for each segment
6. Perform statistical analysis (paired t-tests, effect sizes)
7. Run controls (position, length, tag artifact)
8. Generate visualizations
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
    check_gsm8k_answer,
)
from wor.metrics import (
    calculate_metrics_for_segment,
    paired_t_test,
    fdr_correction,
    interpret_effect_size,
)
from wor.visualization import (
    violin_plot_cud_ape,
    plot_statistical_comparison,
)


def get_attention_states(model, tokenizer, full_text: str):
    """Extract attention weights and hidden states for the full text."""
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
        )
    
    # Stack attention weights: (num_layers, batch, num_heads, seq_len, seq_len)
    attentions = torch.stack(outputs.attentions, dim=0)
    
    # Stack hidden states: (num_layers, batch, seq_len, hidden_size)
    hidden_states = torch.stack(outputs.hidden_states, dim=0)
    
    return attentions, hidden_states, inputs["input_ids"][0]


def find_segment_token_indices(
    tokenizer,
    full_text: str,
    thinking_segment: str,
    response_segment: str,
) -> dict:
    """Find token indices for thinking and response segments."""
    # Tokenize full text
    full_tokens = tokenizer(full_text, add_special_tokens=True, return_tensors="pt")
    token_ids = full_tokens["input_ids"][0]
    
    # Find tag positions
    think_start_tag = "<think>"
    think_end_tag = "</think>"
    
    if think_start_tag in full_text and think_end_tag in full_text:
        tag_start_pos = full_text.find(think_start_tag)
        tag_end_pos = full_text.find(think_end_tag) + len(think_end_tag)
        
        # Tokenize prefix (before thinking)
        prefix = full_text[:tag_start_pos]
        prefix_tokens = tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
        thinking_start_idx = len(prefix_tokens) - 1
        
        # Tokenize up to end of thinking
        up_to_thinking_end = full_text[:tag_end_pos]
        up_to_end_tokens = tokenizer(up_to_thinking_end, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
        thinking_end_idx = len(up_to_end_tokens)
        
        # Response starts after thinking
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


def run_experiment_1_enhanced(
    num_problems: int = 500,
    output_dir: Path = None,
):
    """
    Run enhanced Experiment 1 with statistical analysis and controls.
    
    Args:
        num_problems: Number of GSM8K problems to use (target: 500-1000)
        output_dir: Directory to save results and figures
    """
    if output_dir is None:
        output_dir = Path("results/experiment_1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Experiment 1: The Sparsity Gap (Enhanced)")
    print("=" * 80)
    print()
    
    # Load model
    print("Step 1: Loading model...")
    use_alternative = os.getenv("USE_ALTERNATIVE_LOADER", "false").lower() == "true"
    use_8bit = os.getenv("USE_8BIT_QUANTIZATION", "false").lower() == "true"
    
    # Get model name from env or use default (8B for better research results)
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    print(f"Using model: {model_name}")
    print()
    
    try:
        if use_alternative:
            model, tokenizer = load_deepseek_r1_model_alternative(model_name=model_name, use_8bit=use_8bit)
        else:
            model, tokenizer = load_deepseek_r1_model(model_name=model_name)
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        print("Trying alternative loader...")
        model, tokenizer = load_deepseek_r1_model_alternative(model_name=model_name, use_8bit=True)
        model.eval()
    print()
    
    # Load dataset
    print(f"Step 2: Loading GSM8K dataset ({num_problems} problems)...")
    try:
        problems = load_gsm8k_dataset(num_problems=num_problems, split="test")
        print(f"✓ Loaded {len(problems)} problems")
    except Exception as e:
        print(f"⚠ Dataset loading failed: {e}")
        print("Using fallback problems...")
        from wor.data.datasets import get_gsm8k_fallback_problems
        problems = get_gsm8k_fallback_problems(num_problems)
    print()
    
    # Storage for results
    all_results = []
    correct_results = []
    incorrect_results = []
    
    print("Step 3: Running inference and extracting metrics...")
    print()
    
    for i, problem in enumerate(tqdm(problems, desc="Processing problems")):
        prompt = problem['question']
        correct_answer = problem.get('answer', '')
        
        try:
            # Generate output
            generated = generate_with_reasoning(model, tokenizer, prompt)
            full_text = prompt + generated
            
            # Parse segments
            parsed = parse_reasoning_output(full_text)
            
            if not parsed["has_reasoning"]:
                continue
            
            # Get attention states
            attentions, hidden_states, token_ids = get_attention_states(model, tokenizer, full_text)
            
            # Find token indices
            segment_indices = find_segment_token_indices(
                tokenizer,
                full_text,
                parsed["thinking_segment"],
                parsed["response_segment"],
            )
            
            thinking_start, thinking_end = segment_indices["thinking"]
            response_start, response_end = segment_indices["response"]
            
            # Calculate metrics for thinking segment
            if thinking_end > thinking_start:
                thinking_metrics = calculate_metrics_for_segment(
                    attentions,
                    thinking_start,
                    thinking_end,
                    hidden_states=hidden_states,  # Include hidden states for AE calculation
                )
            else:
                thinking_metrics = {"cud": 0.0, "ape": 0.0, "ae": 0.0, "num_tokens": 0}
            
            # Calculate metrics for response segment
            if response_end > response_start:
                response_metrics = calculate_metrics_for_segment(
                    attentions,
                    response_start,
                    response_end,
                    hidden_states=hidden_states,  # Include hidden states for AE calculation
                )
            else:
                response_metrics = {"cud": 0.0, "ape": 0.0, "ae": 0.0, "num_tokens": 0}
            
            # Check answer correctness
            is_correct, extracted_answer = check_gsm8k_answer(generated, correct_answer)
            
            # Store results
            result = {
                "problem_id": i,
                "question": prompt,
                "correct_answer": correct_answer,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
                "thinking_cud": thinking_metrics["cud"],
                "thinking_ape": thinking_metrics["ape"],
                "thinking_ae": thinking_metrics["ae"],
                "thinking_tokens": thinking_metrics["num_tokens"],
                "response_cud": response_metrics["cud"],
                "response_ape": response_metrics["ape"],
                "response_ae": response_metrics["ae"],
                "response_tokens": response_metrics["num_tokens"],
            }
            
            all_results.append(result)
            
            if is_correct:
                correct_results.append(result)
            else:
                incorrect_results.append(result)
            
        except Exception as e:
            print(f"\n⚠ Error processing problem {i}: {e}")
            continue
    
    print(f"\n✓ Processed {len(all_results)} problems successfully")
    print(f"  Correct: {len(correct_results)}, Incorrect: {len(incorrect_results)}")
    print()
    
    # Statistical Analysis
    print("Step 4: Statistical Analysis")
    print("-" * 80)
    
    if len(all_results) < 2:
        print("⚠ Insufficient data for statistical analysis")
        return
    
    # Extract metric arrays
    thinking_cud = np.array([r["thinking_cud"] for r in all_results])
    response_cud = np.array([r["response_cud"] for r in all_results])
    thinking_ape = np.array([r["thinking_ape"] for r in all_results])
    response_ape = np.array([r["response_ape"] for r in all_results])
    
    # Paired t-tests
    print("\n4.1 Paired t-tests (Thinking vs. Response within same problem):")
    print()
    
    cud_test = paired_t_test(thinking_cud, response_cud)
    ape_test = paired_t_test(thinking_ape, response_ape)
    
    # Apply FDR correction
    p_values = [cud_test['p_value'], ape_test['p_value']]
    corrected_p, rejected = fdr_correction(p_values, alpha=0.05)
    
    cud_test['p_value_corrected'] = corrected_p[0]
    cud_test['rejected'] = rejected[0]
    ape_test['p_value_corrected'] = corrected_p[1]
    ape_test['rejected'] = rejected[1]
    
    # Print results
    print("CUD (Circuit Utilization Density):")
    print(f"  Mean difference: {cud_test['mean_diff']:.4f}")
    print(f"  95% CI: [{cud_test['ci_lower']:.4f}, {cud_test['ci_upper']:.4f}]")
    print(f"  t({cud_test['n']-1}) = {cud_test['t_statistic']:.4f}")
    print(f"  p = {cud_test['p_value']:.4f}")
    print(f"  p (FDR-corrected) = {cud_test['p_value_corrected']:.4f}")
    print(f"  Cohen's d = {cud_test['cohens_d']:.4f} ({interpret_effect_size(cud_test['cohens_d'])} effect)")
    print(f"  Significant: {'Yes' if cud_test['rejected'] else 'No'}")
    print()
    
    print("APE (Attention Process Entropy):")
    print(f"  Mean difference: {ape_test['mean_diff']:.4f}")
    print(f"  95% CI: [{ape_test['ci_lower']:.4f}, {ape_test['ci_upper']:.4f}]")
    print(f"  t({ape_test['n']-1}) = {ape_test['t_statistic']:.4f}")
    print(f"  p = {ape_test['p_value']:.4f}")
    print(f"  p (FDR-corrected) = {ape_test['p_value_corrected']:.4f}")
    print(f"  Cohen's d = {ape_test['cohens_d']:.4f} ({interpret_effect_size(ape_test['cohens_d'])} effect)")
    print(f"  Significant: {'Yes' if ape_test['rejected'] else 'No'}")
    print()
    
    # Summary table
    print("4.2 Summary Statistics:")
    print()
    table_data = [
        ["Metric", "Segment", "Mean", "Std", "Min", "Max"],
        ["CUD", "Thinking", f"{np.mean(thinking_cud):.4f}", f"{np.std(thinking_cud):.4f}",
         f"{np.min(thinking_cud):.4f}", f"{np.max(thinking_cud):.4f}"],
        ["CUD", "Response", f"{np.mean(response_cud):.4f}", f"{np.std(response_cud):.4f}",
         f"{np.min(response_cud):.4f}", f"{np.max(response_cud):.4f}"],
        ["APE", "Thinking", f"{np.mean(thinking_ape):.4f}", f"{np.std(thinking_ape):.4f}",
         f"{np.min(thinking_ape):.4f}", f"{np.max(thinking_ape):.4f}"],
        ["APE", "Response", f"{np.mean(response_ape):.4f}", f"{np.std(response_ape):.4f}",
         f"{np.min(response_ape):.4f}", f"{np.max(response_ape):.4f}"],
    ]
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    print()
    
    # Controls (simplified - full implementation would require more analysis)
    print("4.3 Controls:")
    print("  Note: Full position/length/tag artifact controls require additional analysis")
    print("  See experiment logs for detailed segment information")
    print()
    
    # Visualizations
    print("Step 5: Generating visualizations...")
    thinking_metrics_list = [
        {"cud": r["thinking_cud"], "ape": r["thinking_ape"]} for r in all_results
    ]
    response_metrics_list = [
        {"cud": r["response_cud"], "ape": r["response_ape"]} for r in all_results
    ]
    
    violin_plot_cud_ape(
        thinking_metrics_list,
        response_metrics_list,
        output_path=output_dir / "violin_plot_cud.png",
        metric_name="CUD",
    )
    
    violin_plot_cud_ape(
        thinking_metrics_list,
        response_metrics_list,
        output_path=output_dir / "violin_plot_ape.png",
        metric_name="APE",
    )
    
    # Statistical comparison plot
    plot_statistical_comparison(
        {
            'groups': ['Thinking', 'Response'],
            'means': [np.mean(thinking_cud), np.mean(response_cud)],
            'ci_lower': [cud_test['ci_lower'], 0],  # Simplified
            'ci_upper': [cud_test['ci_upper'], 0],
            'cohens_d': cud_test['cohens_d'],
            'effect_size_label': interpret_effect_size(cud_test['cohens_d']),
        },
        output_path=output_dir / "statistical_comparison.png",
    )
    
    print("✓ Visualizations saved")
    print()
    
    # Save results
    print("Step 6: Saving results...")
    results_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'Experiment 1: The Sparsity Gap (Enhanced)',
            'num_problems': len(all_results),
            'statistical_tests': {
                'cud': cud_test,
                'ape': ape_test,
            },
            'summary_stats': {
                'thinking_cud': {
                    'mean': float(np.mean(thinking_cud)),
                    'std': float(np.std(thinking_cud)),
                },
                'response_cud': {
                    'mean': float(np.mean(response_cud)),
                    'std': float(np.std(response_cud)),
                },
            },
            'results': all_results,
        }, f, indent=2)
    
    print(f"✓ Results saved to {results_file}")
    print()
    
    # Final summary
    print("=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    print()
    print("Key Findings:")
    print(f"  - Thinking segments have {'lower' if cud_test['mean_diff'] < 0 else 'higher'} CUD")
    print(f"  - Effect size: {interpret_effect_size(cud_test['cohens_d'])} (d = {cud_test['cohens_d']:.4f})")
    print(f"  - Statistical significance: {'Yes' if cud_test['rejected'] else 'No'} (p = {cud_test['p_value_corrected']:.4f})")
    print()
    print(f"Results and figures saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run enhanced Experiment 1")
    parser.add_argument("--num-problems", type=int, default=500,
                       help="Number of GSM8K problems to use (default: 500)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results (default: results/experiment_1)")
    
    args = parser.parse_args()
    
    # Validate num_problems
    if args.num_problems <= 0:
        parser.error("--num-problems must be a positive integer")
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_experiment_1_enhanced(
        num_problems=args.num_problems,
        output_dir=output_dir,
    )

