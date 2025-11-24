"""
Statistical analysis functions for research experiments.

Implements t-tests, effect sizes, and multiple comparison correction
as specified in the enhanced research plan.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, List
import warnings


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    
    Cohen's d = (mean1 - mean2) / pooled_std
    
    Interpretation:
    - d = 0.2: Small effect
    - d = 0.5: Medium effect
    - d = 0.8: Large effect
    
    Args:
        group1: First group of values
        group2: Second group of values
        
    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    d = (mean1 - mean2) / pooled_std
    return float(d)


def paired_t_test(group1: np.ndarray, group2: np.ndarray) -> Dict[str, float]:
    """
    Perform paired t-test (for within-subject comparisons).
    
    Used for comparing thinking vs. response segments within the same problem.
    
    Args:
        group1: First group (e.g., thinking segment metrics)
        group2: Second group (e.g., response segment metrics)
        
    Returns:
        Dictionary with:
            - 't_statistic': t-statistic
            - 'p_value': p-value
            - 'cohens_d': Effect size
            - 'mean_diff': Mean difference (group1 - group2)
            - 'ci_lower': Lower bound of 95% confidence interval
            - 'ci_upper': Upper bound of 95% confidence interval
    """
    if len(group1) != len(group2):
        raise ValueError("Groups must have the same length for paired t-test")
    
    # Calculate differences
    differences = group1 - group2
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(group1, group2)
    
    # Calculate effect size
    d = cohens_d(group1, group2)
    
    # Calculate confidence interval for mean difference
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    n = len(differences)
    se = std_diff / np.sqrt(n)
    ci = stats.t.interval(0.95, n - 1, loc=mean_diff, scale=se)
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': d,
        'mean_diff': float(mean_diff),
        'ci_lower': float(ci[0]),
        'ci_upper': float(ci[1]),
        'n': n,
    }


def independent_t_test(group1: np.ndarray, group2: np.ndarray) -> Dict[str, float]:
    """
    Perform independent samples t-test (for between-group comparisons).
    
    Used for comparing correct vs. incorrect reasoning traces.
    
    Args:
        group1: First group (e.g., correct reasoning metrics)
        group2: Second group (e.g., incorrect reasoning metrics)
        
    Returns:
        Dictionary with:
            - 't_statistic': t-statistic
            - 'p_value': p-value
            - 'cohens_d': Effect size
            - 'mean_diff': Mean difference (group1 - group2)
            - 'ci_lower': Lower bound of 95% confidence interval
            - 'ci_upper': Upper bound of 95% confidence interval
    """
    # Perform independent samples t-test
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    # Calculate effect size
    d = cohens_d(group1, group2)
    
    # Calculate confidence interval for mean difference
    mean1, mean2 = np.mean(group1), np.mean(group2)
    mean_diff = mean1 - mean2
    
    # Standard error for difference of means
    se1 = np.std(group1, ddof=1) / np.sqrt(len(group1))
    se2 = np.std(group2, ddof=1) / np.sqrt(len(group2))
    se_diff = np.sqrt(se1**2 + se2**2)
    
    # Degrees of freedom for Welch's t-test approximation
    df = len(group1) + len(group2) - 2
    ci = stats.t.interval(0.95, df, loc=mean_diff, scale=se_diff)
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': d,
        'mean_diff': float(mean_diff),
        'ci_lower': float(ci[0]),
        'ci_upper': float(ci[1]),
        'n1': len(group1),
        'n2': len(group2),
    }


def fdr_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply False Discovery Rate (FDR) correction for multiple comparisons.
    
    Uses Benjamini-Hochberg procedure.
    
    Args:
        p_values: List of p-values to correct
        alpha: Significance level (default 0.05)
        
    Returns:
        Tuple of (corrected_p_values, rejected_hypotheses)
        - corrected_p_values: FDR-corrected p-values
        - rejected_hypotheses: Boolean array indicating which hypotheses are rejected
    """
    p_array = np.array(p_values)
    n = len(p_array)
    
    # Sort p-values
    sorted_indices = np.argsort(p_array)
    sorted_p = p_array[sorted_indices]
    
    # Calculate adjusted p-values (Benjamini-Hochberg)
    adjusted_p = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted_p[sorted_indices[i]] = sorted_p[i]
        else:
            adjusted_p[sorted_indices[i]] = min(
                sorted_p[i] * n / (i + 1),
                adjusted_p[sorted_indices[i + 1]]
            )
    
    # Determine which hypotheses are rejected
    rejected = adjusted_p <= alpha
    
    return adjusted_p, rejected


def two_way_anova(
    data: np.ndarray,
    factor1: np.ndarray,
    factor2: np.ndarray,
) -> Dict[str, float]:
    """
    Perform two-way ANOVA (for task × model comparisons).
    
    Args:
        data: Dependent variable (e.g., CUD values)
        factor1: First factor (e.g., task type: math vs. history)
        factor2: Second factor (e.g., model: DeepSeek vs. Llama)
        
    Returns:
        Dictionary with ANOVA results:
            - 'f_factor1': F-statistic for factor1
            - 'p_factor1': p-value for factor1
            - 'f_factor2': F-statistic for factor2
            - 'p_factor2': p-value for factor2
            - 'f_interaction': F-statistic for interaction
            - 'p_interaction': p-value for interaction
    """
    # Convert factors to numeric if needed
    if factor1.dtype == object:
        factor1_unique = np.unique(factor1)
        factor1_numeric = np.array([np.where(factor1_unique == f)[0][0] for f in factor1])
    else:
        factor1_numeric = factor1
    
    if factor2.dtype == object:
        factor2_unique = np.unique(factor2)
        factor2_numeric = np.array([np.where(factor2_unique == f)[0][0] for f in factor2])
    else:
        factor2_numeric = factor2
    
    # Perform two-way ANOVA
    # Using scipy's f_oneway for simplicity (manual calculation would be more accurate)
    # For proper two-way ANOVA, consider using statsmodels
    try:
        from statsmodels.stats.anova import anova_lm
        from statsmodels.formula.api import ols
        import pandas as pd
        
        df = pd.DataFrame({
            'data': data,
            'factor1': factor1_numeric,
            'factor2': factor2_numeric,
        })
        
        model = ols('data ~ C(factor1) + C(factor2) + C(factor1):C(factor2)', data=df).fit()
        anova_table = anova_lm(model, typ=2)
        
        return {
            'f_factor1': float(anova_table.loc['C(factor1)', 'F']),
            'p_factor1': float(anova_table.loc['C(factor1)', 'PR(>F)']),
            'f_factor2': float(anova_table.loc['C(factor2)', 'F']),
            'p_factor2': float(anova_table.loc['C(factor2)', 'PR(>F)']),
            'f_interaction': float(anova_table.loc['C(factor1):C(factor2)', 'F']),
            'p_interaction': float(anova_table.loc['C(factor1):C(factor2)', 'PR(>F)']),
        }
    except ImportError:
        warnings.warn("statsmodels not available, using simplified ANOVA")
        # Fallback: simple one-way ANOVAs
        groups1 = [data[factor1_numeric == i] for i in np.unique(factor1_numeric)]
        groups2 = [data[factor2_numeric == i] for i in np.unique(factor2_numeric)]
        
        f1, p1 = stats.f_oneway(*groups1)
        f2, p2 = stats.f_oneway(*groups2)
        
        return {
            'f_factor1': float(f1),
            'p_factor1': float(p1),
            'f_factor2': float(f2),
            'p_factor2': float(p2),
            'f_interaction': None,
            'p_interaction': None,
        }


def interpret_effect_size(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

