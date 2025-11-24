# Pre-Registration: The Efficiency of Reason

**Date:** [To be filled]
**Researcher:** [To be filled]
**Project:** Quantifying the Computational Sparsity of Chain-of-Thought in Large Language Models

## 1. Research Question

**Primary Hypothesis:** Reasoning traces (inside `<think>` tags) are computationally sparser than standard text generation, as measured by lower Circuit Utilization Density (CUD) and Attention Process Entropy (APE).

**Alternative Hypotheses to Test:**
- H1: Sparsity is due to the literal `<think>` token embedding, not reasoning content
- H2: Sparsity is a position effect (early vs. late in generation)
- H3: Sparsity is a length effect (thinking segments may be systematically different lengths)

## 2. Planned Analyses

### Experiment 1: The Sparsity Gap

**Primary Analysis:**
- Paired t-test comparing CUD/APE in thinking vs. response segments within the same problem
- Effect size: Cohen's d
- Significance threshold: p < 0.05 (with FDR correction for multiple comparisons)
- Expected effect size: d > 0.5 (medium effect)

**Controls:**
- Position control: Compare early vs. late tokens in thinking segment
- Length control: Compare short vs. long thinking segments
- Tag artifact control: Compare same content with and without `<think>` tags

**Sample Size:**
- Target: 500-1000 problems
- Power analysis: 80% power to detect d = 0.5 requires ~500 problems

### Experiment 1.5: Reasoning Quality Correlation

**Primary Analysis:**
- Independent samples t-test comparing CUD/APE between correct and incorrect reasoning traces
- Hypothesis: Correct reasoning has lower CUD (higher sparsity)
- Significance threshold: p < 0.05

### Experiment 2: Task Contrast

**Primary Analysis:**
- Two-way ANOVA (task × model)
- Factors: Task type (Math vs. History), Model (DeepSeek-R1 vs. Llama-3)
- Significance threshold: p < 0.05

### Experiment 3: Kanan Validation

**Primary Analysis:**
- Compare math accuracy and language fluency before/after ablation
- Expected: Math accuracy decreases significantly, language fluency preserved
- Control: Random head ablation should affect both tasks equally

### Experiment 4: Model Generalization

**Primary Analysis:**
- Meta-analysis of effect sizes across models
- Consistency check: Effect sizes should be similar across models (std < 0.2)

## 3. Exclusion Criteria

- Problems where model output does not contain `<think>` tags
- Problems where tokenization fails
- Problems where attention extraction fails

## 4. Statistical Tests

- **Paired t-tests:** For within-problem comparisons (Experiment 1)
- **Independent t-tests:** For between-group comparisons (Experiment 1.5)
- **Two-way ANOVA:** For factorial designs (Experiment 2)
- **FDR correction:** For multiple comparisons (Benjamini-Hochberg procedure)

## 5. Effect Size Reporting

All analyses will report:
- Cohen's d with interpretation (small/medium/large)
- 95% confidence intervals
- Mean differences with standard deviations

## 6. Visualization Plan

- Violin plots: Distribution of CUD/APE by condition
- Scatter plots: Quality correlation with regression lines
- Heatmaps: Attention head activation patterns
- Statistical comparison plots: Means with confidence intervals

## 7. Data Sharing

- All attention activations and metrics will be saved
- Code for reproducing all analyses will be provided
- Results will be anonymized if needed

## 8. Deviations

Any deviations from this pre-registration will be documented and justified in the final paper.

