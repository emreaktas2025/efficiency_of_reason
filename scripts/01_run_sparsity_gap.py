#!/usr/bin/env python3
"""
Experiment 1: The Sparsity Gap

This script tests the hypothesis that reasoning traces (inside <think> tags)
are computationally sparser than standard text generation.

Note: DeepSeek-R1 actually uses <think> tags. The research plan mentions
<think> as a conceptual placeholder, but the model outputs <think> tags.

Procedure:
1. Load DeepSeek-R1 model with 4-bit quantization
2. Run inference on GSM8K-style math problems
3. Extract internal attention states
4. Parse outputs into thinking vs response segments
5. Calculate CUD and APE metrics for each segment
6. Compare sparsity between thinking and response phases
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from transformers import GenerationConfig
from tabulate import tabulate

from wor.core import load_deepseek_r1_model
from wor.data import parse_reasoning_output, get_token_indices_for_segments
from wor.metrics import calculate_metrics_for_segment


def get_gsm8k_prompts() -> list[str]:
    """Return a small set of hard GSM8K-style math problems."""
    return [
        "Solve this step by step: A store has 15 boxes of apples. Each box contains 8 apples. "
        "If 3 boxes are sold and 2 more boxes are added, how many apples are in the store now?",
        
        "A train travels 120 miles in 2 hours. Then it travels another 180 miles in 3 hours. "
        "What is the average speed of the train for the entire journey?",
        
        "Sarah has 24 stickers. She gives away 1/3 of them to her friend. Then she buys 10 more stickers. "
        "How many stickers does Sarah have now?",
        
        "A rectangle has a length of 12 cm and a width of 8 cm. If the length is increased by 25% "
        "and the width is decreased by 20%, what is the new area of the rectangle?",
        
        "Tom reads 15 pages of a book on Monday, 20 pages on Tuesday, and 25 pages on Wednesday. "
        "If the book has 180 pages total, how many pages does Tom need to read to finish the book?",
    ]


def generate_with_reasoning(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """
    Generate text with the model, allowing it to produce reasoning traces.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Generated text (including reasoning)
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with reasoning
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Remove the input prompt from the output
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text


def get_attention_states(model, tokenizer, full_text: str):
    """
    Run forward pass to extract attention weights for the full generated text.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        full_text: Complete text (prompt + generated)
        
    Returns:
        Tuple of (attention_weights, token_ids)
        attention_weights shape: (num_layers, batch, num_heads, seq_len, seq_len)
    """
    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    
    # Forward pass with attention output
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
        )
    
    # Extract attention weights
    # outputs.attentions is a tuple of tensors, one per layer
    # Each tensor shape: (batch, num_heads, seq_len, seq_len)
    attentions = torch.stack(outputs.attentions, dim=0)  # (num_layers, batch, num_heads, seq_len, seq_len)
    
    return attentions, inputs["input_ids"][0]


def find_segment_token_indices(
    tokenizer,
    full_text: str,
    thinking_segment: str,
    response_segment: str,
) -> dict:
    """
    Find token indices for thinking and response segments.
    
    This is a more robust method that tokenizes the full text and
    searches for segment boundaries.
    """
    # Tokenize full text
    full_tokens = tokenizer(full_text, add_special_tokens=True, return_tensors="pt")
    token_ids = full_tokens["input_ids"][0]
    decoded_full = tokenizer.decode(token_ids, skip_special_tokens=False)
    
    # Find tag positions in original text
    # DeepSeek-R1 uses <think> tags (the model's actual output format)
    think_start_tag = "<think>"
    think_end_tag = "</think>"
    
    if think_start_tag in full_text and think_end_tag in full_text:
        # Find positions
        tag_start_pos = full_text.find(think_start_tag)
        tag_end_pos = full_text.find(think_end_tag) + len(think_end_tag)
        
        # Tokenize prefix (before thinking)
        prefix = full_text[:tag_start_pos]
        prefix_tokens = tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
        thinking_start_idx = len(prefix_tokens) - 1  # -1 because we'll include the start token
        
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
        # No reasoning tags - entire output is response
        return {
            "thinking": (0, 0),
            "response": (0, len(token_ids)),
        }


def run_experiment():
    """Main experiment execution."""
    print("=" * 80)
    print("Experiment 1: The Sparsity Gap")
    print("=" * 80)
    print()
    
    # Load model
    print("Step 1: Loading model...")
    model, tokenizer = load_deepseek_r1_model()
    model.eval()
    print()
    
    # Get prompts
    prompts = get_gsm8k_prompts()
    print(f"Step 2: Running inference on {len(prompts)} prompts...")
    print()
    
    # Storage for results
    all_results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"Processing prompt {i}/{len(prompts)}...")
        print(f"Prompt: {prompt[:80]}...")
        
        try:
            # Generate output
            generated = generate_with_reasoning(model, tokenizer, prompt)
            full_text = prompt + generated
            
            # Parse segments
            parsed = parse_reasoning_output(full_text)
            
            if not parsed["has_reasoning"]:
                print(f"  Warning: No reasoning tags found in output for prompt {i}")
                print(f"  Generated text: {generated[:200]}...")
                print()
                continue
            
            print(f"  Found reasoning segment ({len(parsed['thinking_segment'])} chars)")
            print(f"  Found response segment ({len(parsed['response_segment'])} chars)")
            
            # Get attention states
            attentions, token_ids = get_attention_states(model, tokenizer, full_text)
            
            # Find token indices for segments
            segment_indices = find_segment_token_indices(
                tokenizer,
                full_text,
                parsed["thinking_segment"],
                parsed["response_segment"],
            )
            
            thinking_start, thinking_end = segment_indices["thinking"]
            response_start, response_end = segment_indices["response"]
            
            print(f"  Thinking tokens: {thinking_start} to {thinking_end} ({thinking_end - thinking_start} tokens)")
            print(f"  Response tokens: {response_start} to {response_end} ({response_end - response_start} tokens)")
            
            # Calculate metrics for thinking segment
            if thinking_end > thinking_start:
                thinking_metrics = calculate_metrics_for_segment(
                    attentions,
                    thinking_start,
                    thinking_end,
                    threshold=1e-3,
                )
            else:
                thinking_metrics = {"cud": 0.0, "ape": 0.0, "num_tokens": 0}
            
            # Calculate metrics for response segment
            if response_end > response_start:
                response_metrics = calculate_metrics_for_segment(
                    attentions,
                    response_start,
                    response_end,
                    threshold=1e-3,
                )
            else:
                response_metrics = {"cud": 0.0, "ape": 0.0, "num_tokens": 0}
            
            # Store results
            all_results.append({
                "prompt_id": i,
                "thinking_cud": thinking_metrics["cud"],
                "thinking_ape": thinking_metrics["ape"],
                "thinking_tokens": thinking_metrics["num_tokens"],
                "response_cud": response_metrics["cud"],
                "response_ape": response_metrics["ape"],
                "response_tokens": response_metrics["num_tokens"],
            })
            
            print(f"  Thinking CUD: {thinking_metrics['cud']:.4f}, APE: {thinking_metrics['ape']:.4f}")
            print(f"  Response CUD: {response_metrics['cud']:.4f}, APE: {response_metrics['ape']:.4f}")
            print()
            
        except Exception as e:
            print(f"  Error processing prompt {i}: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue
    
    # Print summary table
    if all_results:
        print("=" * 80)
        print("Summary Results")
        print("=" * 80)
        print()
        
        # Calculate averages
        avg_thinking_cud = sum(r["thinking_cud"] for r in all_results) / len(all_results)
        avg_thinking_ape = sum(r["thinking_ape"] for r in all_results) / len(all_results)
        avg_response_cud = sum(r["response_cud"] for r in all_results) / len(all_results)
        avg_response_ape = sum(r["response_ape"] for r in all_results) / len(all_results)
        
        # Create comparison table
        table_data = [
            ["Segment", "CUD (Circuit Utilization Density)", "APE (Attention Process Entropy)", "Avg Tokens"],
            ["Thinking", f"{avg_thinking_cud:.4f}", f"{avg_thinking_ape:.4f}", 
             f"{sum(r['thinking_tokens'] for r in all_results) / len(all_results):.1f}"],
            ["Response", f"{avg_response_cud:.4f}", f"{avg_response_ape:.4f}",
             f"{sum(r['response_tokens'] for r in all_results) / len(all_results):.1f}"],
            ["Difference", f"{avg_thinking_cud - avg_response_cud:.4f}", 
             f"{avg_thinking_ape - avg_response_ape:.4f}", "-"],
        ]
        
        print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
        print()
        
        # Hypothesis test
        print("Hypothesis Test:")
        print(f"  Lower CUD in thinking = Higher sparsity = More focused computation")
        print(f"  Thinking CUD: {avg_thinking_cud:.4f}")
        print(f"  Response CUD: {avg_response_cud:.4f}")
        
        if avg_thinking_cud < avg_response_cud:
            print(f"  ✓ Hypothesis SUPPORTED: Thinking is sparser (CUD difference: {avg_response_cud - avg_thinking_cud:.4f})")
        else:
            print(f"  ✗ Hypothesis NOT SUPPORTED: Thinking is not sparser (CUD difference: {avg_thinking_cud - avg_response_cud:.4f})")
        
        print()
        print(f"  Lower APE in thinking = More focused attention = More structured flow")
        print(f"  Thinking APE: {avg_thinking_ape:.4f}")
        print(f"  Response APE: {avg_response_ape:.4f}")
        
        if avg_thinking_ape < avg_response_ape:
            print(f"  ✓ Hypothesis SUPPORTED: Thinking has more focused attention (APE difference: {avg_response_ape - avg_thinking_ape:.4f})")
        else:
            print(f"  ✗ Hypothesis NOT SUPPORTED: Thinking does not have more focused attention (APE difference: {avg_thinking_ape - avg_response_ape:.4f})")
    else:
        print("No successful results to summarize.")
    
    print()
    print("Experiment complete!")


if __name__ == "__main__":
    run_experiment()

