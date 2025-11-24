#!/usr/bin/env python3
"""
Experiment 1.5: Reasoning Quality Correlation

Tests whether sparsity correlates with reasoning quality.
Does correct reasoning show higher sparsity (lower CUD) than incorrect reasoning?

Procedure:
1. Use results from Experiment 1 (or run new analysis)
2. Classify outputs as "Correct" or "Incorrect" based on final answer
3. Compare CUD/APE metrics between correct and incorrect reasoning traces
4. Statistical test: Independent samples t-test
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

from wor.metrics import (
    independent_t_test,
    interpret_effect_size,
)
from wor.visualization import (
    scatter_plot_quality_correlation,
    plot_statistical_comparison,
)


def analyze_quality_correlation(
    results_file: Path,
    output_dir: Path = None,
):
    """
    Analyze correlation between reasoning quality and sparsity metrics.
    
    Args:
        results_file: Path to Experiment 1 results JSON file
        output_dir: Directory to save analysis results
    """
    if output_dir is None:
        output_dir = Path("results/experiment_1.5")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Experiment 1.5: Reasoning Quality Correlation")
    print("=" * 80)
    print()
    
    # Load results from Experiment 1
    print(f"Loading results from: {results_file}")
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    print(f"✓ Loaded {len(results)} results")
    print()
    
    # Separate correct and incorrect
    correct_results = [r for r in results if r.get('is_correct', False)]
    incorrect_results = [r for r in results if not r.get('is_correct', False)]
    
    print(f"Correct reasoning traces: {len(correct_results)}")
    print(f"Incorrect reasoning traces: {len(incorrect_results)}")
    print()
    
    if len(correct_results) < 2 or len(incorrect_results) < 2:
        print("⚠ Insufficient data for statistical analysis")
        print("Need at least 2 correct and 2 incorrect traces")
        return
    
    # Extract metrics
    correct_cud = np.array([r["thinking_cud"] for r in correct_results])
    incorrect_cud = np.array([r["thinking_cud"] for r in incorrect_results])
    correct_ape = np.array([r["thinking_ape"] for r in correct_results])
    incorrect_ape = np.array([r["thinking_ape"] for r in incorrect_results])
    
    # Statistical Analysis
    print("Statistical Analysis:")
    print("-" * 80)
    print()
    
    print("CUD (Circuit Utilization Density):")
    print("  Hypothesis: Correct reasoning has LOWER CUD (higher sparsity)")
    print()
    
    cud_test = independent_t_test(correct_cud, incorrect_cud)
    
    print(f"  Correct mean: {np.mean(correct_cud):.4f} ± {np.std(correct_cud):.4f}")
    print(f"  Incorrect mean: {np.mean(incorrect_cud):.4f} ± {np.std(incorrect_cud):.4f}")
    print(f"  Mean difference: {cud_test['mean_diff']:.4f}")
    print(f"  95% CI: [{cud_test['ci_lower']:.4f}, {cud_test['ci_upper']:.4f}]")
    print(f"  t({cud_test['n1'] + cud_test['n2'] - 2}) = {cud_test['t_statistic']:.4f}")
    print(f"  p = {cud_test['p_value']:.4f}")
    print(f"  Cohen's d = {cud_test['cohens_d']:.4f} ({interpret_effect_size(cud_test['cohens_d'])} effect)")
    
    if cud_test['p_value'] < 0.05:
        direction = "lower" if cud_test['mean_diff'] < 0 else "higher"
        print(f"  ✓ Significant: Correct reasoning has {direction} CUD")
    else:
        print(f"  ✗ Not significant")
    print()
    
    print("APE (Attention Process Entropy):")
    print("  Hypothesis: Correct reasoning has LOWER APE (more focused attention)")
    print()
    
    ape_test = independent_t_test(correct_ape, incorrect_ape)
    
    print(f"  Correct mean: {np.mean(correct_ape):.4f} ± {np.std(correct_ape):.4f}")
    print(f"  Incorrect mean: {np.mean(incorrect_ape):.4f} ± {np.std(incorrect_ape):.4f}")
    print(f"  Mean difference: {ape_test['mean_diff']:.4f}")
    print(f"  95% CI: [{ape_test['ci_lower']:.4f}, {ape_test['ci_upper']:.4f}]")
    print(f"  t({ape_test['n1'] + ape_test['n2'] - 2}) = {ape_test['t_statistic']:.4f}")
    print(f"  p = {ape_test['p_value']:.4f}")
    print(f"  Cohen's d = {ape_test['cohens_d']:.4f} ({interpret_effect_size(ape_test['cohens_d'])} effect)")
    
    if ape_test['p_value'] < 0.05:
        direction = "lower" if ape_test['mean_diff'] < 0 else "higher"
        print(f"  ✓ Significant: Correct reasoning has {direction} APE")
    else:
        print(f"  ✗ Not significant")
    print()
    
    # Summary table
    print("Summary Table:")
    print()
    table_data = [
        ["Metric", "Quality", "Mean", "Std", "N"],
        ["CUD", "Correct", f"{np.mean(correct_cud):.4f}", f"{np.std(correct_cud):.4f}", len(correct_cud)],
        ["CUD", "Incorrect", f"{np.mean(incorrect_cud):.4f}", f"{np.std(incorrect_cud):.4f}", len(incorrect_cud)],
        ["APE", "Correct", f"{np.mean(correct_ape):.4f}", f"{np.std(correct_ape):.4f}", len(correct_ape)],
        ["APE", "Incorrect", f"{np.mean(incorrect_ape):.4f}", f"{np.std(incorrect_ape):.4f}", len(incorrect_ape)],
    ]
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    print()
    
    # Visualizations
    print("Generating visualizations...")
    
    correct_metrics_list = [
        {"cud": r["thinking_cud"], "ape": r["thinking_ape"]} for r in correct_results
    ]
    incorrect_metrics_list = [
        {"cud": r["thinking_cud"], "ape": r["thinking_ape"]} for r in incorrect_results
    ]
    
    scatter_plot_quality_correlation(
        correct_metrics_list,
        incorrect_metrics_list,
        output_path=output_dir / "quality_correlation_cud.png",
        metric_name="CUD",
    )
    
    scatter_plot_quality_correlation(
        correct_metrics_list,
        incorrect_metrics_list,
        output_path=output_dir / "quality_correlation_ape.png",
        metric_name="APE",
    )
    
    # Statistical comparison plot
    plot_statistical_comparison(
        {
            'groups': ['Correct', 'Incorrect'],
            'means': [np.mean(correct_cud), np.mean(incorrect_cud)],
            'ci_lower': [cud_test['ci_lower'], 0],
            'ci_upper': [cud_test['ci_upper'], 0],
            'cohens_d': cud_test['cohens_d'],
            'effect_size_label': interpret_effect_size(cud_test['cohens_d']),
        },
        output_path=output_dir / "quality_statistical_comparison.png",
    )
    
    print("✓ Visualizations saved")
    print()
    
    # Save analysis results
    analysis_results = {
        'experiment': 'Experiment 1.5: Reasoning Quality Correlation',
        'statistical_tests': {
            'cud': cud_test,
            'ape': ape_test,
        },
        'summary': {
            'n_correct': len(correct_results),
            'n_incorrect': len(incorrect_results),
            'correct_cud_mean': float(np.mean(correct_cud)),
            'incorrect_cud_mean': float(np.mean(incorrect_cud)),
            'correct_ape_mean': float(np.mean(correct_ape)),
            'incorrect_ape_mean': float(np.mean(incorrect_ape)),
        },
    }
    
    results_file = output_dir / "quality_correlation_analysis.json"
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"✓ Analysis results saved to {results_file}")
    print()
    
    # Final summary
    print("=" * 80)
    print("Key Finding:")
    if cud_test['p_value'] < 0.05 and cud_test['mean_diff'] < 0:
        print("  ✓ CONFIRMED: Correct reasoning shows HIGHER sparsity (lower CUD)")
        print("    This supports the hypothesis: 'Focused thinking → Better answers'")
    elif cud_test['p_value'] < 0.05:
        print("  ⚠ Significant difference found, but in opposite direction")
        print("    Correct reasoning has HIGHER CUD (less sparse)")
    else:
        print("  ✗ No significant difference found")
        print("    Sparsity does not correlate with reasoning quality")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Experiment 1.5: Quality Correlation")
    parser.add_argument("--results-file", type=str, required=True,
                       help="Path to Experiment 1 results JSON file")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    analyze_quality_correlation(results_file, output_dir)

