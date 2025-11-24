"""
Visualization functions for research experiments.

Implements violin plots, scatter plots, and heatmaps as specified
in the enhanced research plan.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


def setup_plot_style():
    """Set up publication-quality plot style."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12


def violin_plot_cud_ape(
    thinking_metrics: List[Dict],
    response_metrics: List[Dict],
    output_path: Optional[Path] = None,
    metric_name: str = "CUD",
) -> None:
    """
    Create violin plot showing distribution of CUD/APE by condition.
    
    Args:
        thinking_metrics: List of metric dictionaries for thinking segments
        response_metrics: List of metric dictionaries for response segments
        output_path: Optional path to save the figure
        metric_name: Name of metric to plot ("CUD" or "APE")
    """
    setup_plot_style()
    
    # Extract metric values
    thinking_values = [m.get(metric_name.lower(), 0) for m in thinking_metrics]
    response_values = [m.get(metric_name.lower(), 0) for m in response_metrics]
    
    # Prepare data for seaborn
    data = []
    labels = []
    data.extend(thinking_values)
    labels.extend(['Thinking'] * len(thinking_values))
    data.extend(response_values)
    labels.extend(['Response'] * len(response_values))
    
    df = {'Value': data, 'Segment': labels}
    
    # Create violin plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(data=df, x='Segment', y='Value', ax=ax, palette='Set2')
    ax.set_ylabel(metric_name)
    ax.set_title(f'Distribution of {metric_name} by Segment Type')
    
    # Add mean markers
    thinking_mean = np.mean(thinking_values)
    response_mean = np.mean(response_values)
    ax.plot([-0.2, 0.2], [thinking_mean, thinking_mean], 'r--', alpha=0.7, label=f'Mean: {thinking_mean:.4f}')
    ax.plot([0.8, 1.2], [response_mean, response_mean], 'r--', alpha=0.7, label=f'Mean: {response_mean:.4f}')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    
    plt.close()


def scatter_plot_quality_correlation(
    correct_metrics: List[Dict],
    incorrect_metrics: List[Dict],
    output_path: Optional[Path] = None,
    metric_name: str = "CUD",
) -> None:
    """
    Create scatter plot with regression line for quality correlation.
    
    Shows relationship between reasoning quality (correct/incorrect) and sparsity metrics.
    
    Args:
        correct_metrics: List of metric dictionaries for correct reasoning traces
        incorrect_metrics: List of metric dictionaries for incorrect reasoning traces
        output_path: Optional path to save the figure
        metric_name: Name of metric to plot ("CUD" or "APE")
    """
    setup_plot_style()
    
    # Extract metric values
    correct_values = [m.get(metric_name.lower(), 0) for m in correct_metrics]
    incorrect_values = [m.get(metric_name.lower(), 0) for m in incorrect_metrics]
    
    # Prepare data
    all_values = correct_values + incorrect_values
    all_labels = ['Correct'] * len(correct_values) + ['Incorrect'] * len(incorrect_values)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot
    x_correct = np.random.normal(0, 0.1, len(correct_values))  # Jitter for visibility
    x_incorrect = np.random.normal(1, 0.1, len(incorrect_values))
    
    ax.scatter(x_correct, correct_values, alpha=0.6, label='Correct', s=50)
    ax.scatter(x_incorrect, incorrect_values, alpha=0.6, label='Incorrect', s=50)
    
    # Add box plots
    positions = [0, 1]
    bp = ax.boxplot([correct_values, incorrect_values], positions=positions, widths=0.3, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.5)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Correct', 'Incorrect'])
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} vs. Reasoning Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    
    plt.close()


def heatmap_attention_patterns(
    attention_weights: np.ndarray,
    layer_indices: Optional[List[int]] = None,
    head_indices: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
    title: str = "Attention Head Activation Patterns",
) -> None:
    """
    Create heatmap of attention head activation patterns.
    
    Args:
        attention_weights: Attention tensor (num_layers, num_heads, seq_len, seq_len)
                          or aggregated (num_layers, num_heads)
        layer_indices: Optional list of layer indices to include
        head_indices: Optional list of head indices to include
        output_path: Optional path to save the figure
        title: Plot title
    """
    setup_plot_style()
    
    # Aggregate attention if needed (average over sequence dimensions)
    if attention_weights.ndim == 4:
        # (num_layers, num_heads, seq_len, seq_len) -> (num_layers, num_heads)
        attention_weights = attention_weights.mean(axis=(2, 3))
    elif attention_weights.ndim == 3:
        # (num_layers, batch, num_heads) -> (num_layers, num_heads)
        attention_weights = attention_weights.mean(axis=1)
    
    # Select layers and heads if specified
    if layer_indices is not None:
        attention_weights = attention_weights[layer_indices, :]
    if head_indices is not None:
        attention_weights = attention_weights[:, head_indices]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        cmap='viridis',
        cbar_kws={'label': 'Activation Strength'},
        ax=ax,
    )
    
    ax.set_xlabel('Attention Head')
    ax.set_ylabel('Layer')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_statistical_comparison(
    results: Dict,
    output_path: Optional[Path] = None,
) -> None:
    """
    Create a comprehensive statistical comparison plot.
    
    Shows means, confidence intervals, and effect sizes.
    
    Args:
        results: Dictionary with statistical test results
        output_path: Optional path to save the figure
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Means with confidence intervals
    groups = results.get('groups', ['Group 1', 'Group 2'])
    means = results.get('means', [0, 0])
    ci_lower = results.get('ci_lower', [0, 0])
    ci_upper = results.get('ci_upper', [0, 0])
    
    x_pos = np.arange(len(groups))
    ax1.bar(x_pos, means, yerr=[np.array(means) - np.array(ci_lower), 
                                 np.array(ci_upper) - np.array(means)],
            capsize=10, alpha=0.7, color=['#3498db', '#e74c3c'])
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(groups)
    ax1.set_ylabel('Metric Value')
    ax1.set_title('Mean Comparison with 95% CI')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Effect size visualization
    cohens_d = results.get('cohens_d', 0)
    effect_size_label = results.get('effect_size_label', 'Unknown')
    
    ax2.barh([0], [cohens_d], color='green' if abs(cohens_d) > 0.5 else 'orange')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.axvline(x=0.2, color='gray', linestyle=':', linewidth=1, label='Small')
    ax2.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, label='Medium')
    ax2.axvline(x=0.8, color='gray', linestyle=':', linewidth=1, label='Large')
    ax2.set_xlabel("Cohen's d")
    ax2.set_title(f'Effect Size: {effect_size_label}')
    ax2.set_yticks([])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    
    plt.close()

