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

### Experiment 1: The Sparsity Gap

This experiment tests whether reasoning traces (inside `<think>` tags) have lower CUD than response segments.

Note: DeepSeek-R1 uses `<think>` tags (the model's actual output format).

**If installed as package:**
```bash
python scripts/01_run_sparsity_gap.py
```

**If not installed, run from project root:**
```bash
cd /path/to/efficiency_of_reason
python scripts/01_run_sparsity_gap.py
```

The script will:
1. Load DeepSeek-R1 with 4-bit quantization
2. Run inference on 5 GSM8K-style math problems
3. Extract attention states from forward passes
4. Calculate CUD and APE for thinking vs response segments
5. Display a comparison table

## Metrics

- **CUD (Circuit Utilization Density)**: Percentage of attention heads with activations above threshold. Lower = more sparse.
- **APE (Attention Process Entropy)**: Shannon entropy of attention distributions. Lower = more focused.

## Research Plan

See `RESEARCH_PLAN.md` for the complete research design, methodology, and experimental plan.

