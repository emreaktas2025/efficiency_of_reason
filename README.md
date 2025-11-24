# The Efficiency of Reason

**Quantifying the Computational Sparsity of Chain-of-Thought in Large Language Models**

This repository implements mechanistic interpretability experiments to test the hypothesis that reasoning traces are computationally sparser than standard text generation.

## Overview

This project analyzes DeepSeek-R1-Distill-Llama-8B to measure structural differences between "thinking" (internal reasoning) and "speaking" (text generation) phases using metrics like Circuit Utilization Density (CUD) and Attention Process Entropy (APE).

## Setup

### Requirements

- Python 3.8+
- CUDA-capable GPU (tested on RTX 4090 with 24GB VRAM)
- PyTorch with CUDA support

### Installation

**Option 1: Install as package (Recommended)**
```bash
pip install -r requirements.txt
pip install -e .
```

**Option 2: Install dependencies only**
```bash
pip install -r requirements.txt
# Then run scripts from project root
```

## Project Structure

```
.
├── src/wor/
│   ├── core/
│   │   └── loader.py          # Model loading with 4-bit quantization
│   ├── data/
│   │   └── parser.py           # Parse thinking/response segments
│   └── metrics/
│       └── sparsity.py         # CUD and APE calculations
├── scripts/
│   └── 01_run_sparsity_gap.py  # Experiment 1: The Sparsity Gap
└── RESEARCH_PLAN.md            # Detailed research design
```

## Running Experiments

### Experiment 1: The Sparsity Gap (Enhanced)

**Enhanced version with statistical analysis and increased sample size:**

```bash
python scripts/01_experiment_sparsity_gap_enhanced.py --num-problems 500
```

**Original version (quick test with 5 problems):**

```bash
python scripts/01_run_sparsity_gap.py
```

The enhanced script will:
1. Load DeepSeek-R1 with 4-bit quantization
2. Run inference on 500-1000 GSM8K problems
3. Extract attention states and hidden states
4. Calculate CUD, APE, and AE for thinking vs response segments
5. Perform statistical analysis (paired t-tests, effect sizes, FDR correction)
6. Generate visualizations (violin plots, statistical comparisons)
7. Save comprehensive results to JSON

### Experiment 1.5: Reasoning Quality Correlation

Tests if sparsity correlates with reasoning quality:

```bash
# First run Experiment 1, then:
python scripts/01.5_experiment_quality_correlation.py --results-file results/experiment_1/results_*.json
```

### Experiment 2: Task Contrast

Compares Math vs. History tasks:

```bash
python scripts/02_experiment_task_contrast.py --num-problems 500
```

### Experiment 3: Kanan Validation (Ablation)

Tests modularity through head ablation:

```bash
python scripts/03_experiment_kanan_validation.py --num-problems 100
```

### Experiment 4: Model Generalization

Tests sparsity across multiple models:

```bash
python scripts/04_experiment_model_generalization.py
```

## Metrics

- **CUD (Circuit Utilization Density)**: Percentage of attention heads with activations above threshold. Lower = more sparse.
- **APE (Attention Process Entropy)**: Shannon entropy of attention distributions. Lower = more focused.
- **AE (Activation Energy)**: L2 norm of hidden states. Measures overall activation magnitude.

## Statistical Analysis

All experiments include:
- **Paired t-tests**: For within-problem comparisons
- **Independent t-tests**: For between-group comparisons
- **Two-way ANOVA**: For factorial designs
- **Effect sizes**: Cohen's d with interpretation
- **Multiple comparison correction**: FDR (Benjamini-Hochberg)
- **Confidence intervals**: 95% CIs for all comparisons

## Research Plans

- **`RESEARCH_PLAN.md`**: Original research plan
- **`ENHANCED_RESEARCH_PLAN.md`**: Enhanced plan with statistical rigor and expanded experiments
- **`PREREGISTRATION.md`**: Pre-registered hypotheses and analysis plan

## Enhanced Features

The enhanced implementation includes:
- Statistical analysis with proper tests and effect sizes
- Increased sample sizes (500-1000 problems)
- Quality correlation analysis
- Model generalization testing
- Comprehensive controls (position, length, tag artifact)
- Publication-quality visualizations
- Pre-registration documentation

