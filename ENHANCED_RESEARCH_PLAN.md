# Enhanced Research Plan: The Efficiency of Reason

**Subtitle:** Quantifying the Computational Sparsity of Chain-of-Thought in Large Language Models

This document implements the enhanced research plan with statistical rigor, expanded experiments, and comprehensive controls.

## Overview

This enhanced plan addresses the original research plan with:
- **Statistical rigor**: Proper t-tests, effect sizes, multiple comparison correction
- **Expanded sample sizes**: 500-1000 problems (increased from 100)
- **Additional controls**: Position, length, and tag artifact controls
- **Quality correlation**: Experiment 1.5 testing if sparsity predicts correctness
- **Model generalization**: Testing across multiple models
- **Pre-registration**: Documented hypotheses and analysis plan

## Key Improvements Over Original Plan

1. **Statistical Analysis**: All experiments include proper statistical tests with effect sizes
2. **Increased Power**: Sample sizes increased to 500-1000 for 80% power
3. **Quality Correlation**: New experiment testing if focused thinking → better answers
4. **Baseline Comparison**: Llama-3-8B baseline to control for architecture effects
5. **Model Generalization**: Testing across model sizes for generalizability
6. **Comprehensive Controls**: Position, length, and tag artifact controls

## Experiments

### Experiment 1: The Sparsity Gap (Enhanced)
- **Script**: `scripts/01_experiment_sparsity_gap_enhanced.py`
- **Sample Size**: 500-1000 problems
- **Analysis**: Paired t-tests, effect sizes, FDR correction
- **Controls**: Position, length, tag artifact
- **Output**: Statistical results, visualizations, JSON data

### Experiment 1.5: Reasoning Quality Correlation
- **Script**: `scripts/01.5_experiment_quality_correlation.py`
- **Analysis**: Independent t-tests comparing correct vs. incorrect traces
- **Hypothesis**: Correct reasoning shows higher sparsity
- **Output**: Quality correlation analysis and visualizations

### Experiment 2: Task Contrast
- **Script**: `scripts/02_experiment_task_contrast.py`
- **Tasks**: Math (GSM8K) vs. History (MMLU)
- **Analysis**: Two-way ANOVA (task × model)
- **Baseline**: Llama-3-8B comparison
- **Output**: Task contrast analysis

### Experiment 3: Kanan Validation (Ablation)
- **Script**: `scripts/03_experiment_kanan_validation.py`
- **Method**: Zero-ablate top 1% reasoning heads
- **Tests**: Math performance vs. language fluency
- **Control**: Random head ablation
- **Output**: Ablation results

### Experiment 4: Model Generalization
- **Script**: `scripts/04_experiment_model_generalization.py`
- **Models**: DeepSeek-R1-8B, DeepSeek-R1-1.5B
- **Analysis**: Meta-analysis of effect sizes
- **Output**: Generalization analysis

## Usage

### Running Experiments

```bash
# Experiment 1 (Enhanced)
python scripts/01_experiment_sparsity_gap_enhanced.py --num-problems 500

# Experiment 1.5 (requires Experiment 1 results)
python scripts/01.5_experiment_quality_correlation.py --results-file results/experiment_1/results_*.json

# Experiment 2
python scripts/02_experiment_task_contrast.py --num-problems 500

# Experiment 3
python scripts/03_experiment_kanan_validation.py --num-problems 100

# Experiment 4
python scripts/04_experiment_model_generalization.py
```

## Statistical Analysis

All experiments use:
- **Effect sizes**: Cohen's d with interpretation
- **Confidence intervals**: 95% CIs for all comparisons
- **Multiple comparison correction**: FDR (Benjamini-Hochberg)
- **Power analysis**: 80% power to detect d = 0.5

## Pre-Registration

See `PREREGISTRATION.md` for documented hypotheses and analysis plan.

## Results Structure

All experiments save results in `results/experiment_X/` with:
- JSON files containing all metrics and statistical tests
- Visualization figures (PNG, 300 DPI)
- Summary statistics and interpretations

