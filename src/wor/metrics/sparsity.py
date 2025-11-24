"""
Sparsity metrics for analyzing reasoning traces.

Implements CUD (Circuit Utilization Density) and APE (Attention Process Entropy)
to measure computational sparsity in model activations.
"""

import torch
import torch.nn.functional as F
from typing import Union, Optional


def calculate_cud(
    attention_weights: torch.Tensor,
    threshold: float = 1e-3,
    layer_wise: bool = False,
) -> Union[float, torch.Tensor]:
    """
    Calculate Circuit Utilization Density (CUD).
    
    CUD measures the percentage of attention heads that are "active"
    (have activations above a threshold), indicating sparsity.
    
    Lower CUD = Higher sparsity = More focused computation
    
    Args:
        attention_weights: Tensor of shape (batch, num_heads, seq_len, seq_len)
                          or (num_layers, batch, num_heads, seq_len, seq_len)
        threshold: Activation threshold for considering a head "active"
        layer_wise: If True, return CUD per layer. If False, return average.
        
    Returns:
        CUD value(s) as float or tensor
    """
    if attention_weights.dim() == 5:
        # Multi-layer: (num_layers, batch, num_heads, seq_len, seq_len)
        num_layers, batch, num_heads, seq_len, _ = attention_weights.shape
        
        # Average over sequence dimensions to get per-head activation strength
        head_activations = attention_weights.mean(dim=(-2, -1))  # (num_layers, batch, num_heads)
        
        # Count active heads (above threshold)
        active_heads = (head_activations > threshold).float()
        
        # Calculate percentage of active heads
        total_heads = num_heads
        cud_per_layer = active_heads.mean(dim=(-2, -1))  # Average over batch and heads
        
        if layer_wise:
            return cud_per_layer
        else:
            return cud_per_layer.mean().item()
            
    elif attention_weights.dim() == 4:
        # Single layer: (batch, num_heads, seq_len, seq_len)
        batch, num_heads, seq_len, _ = attention_weights.shape
        
        # Average over sequence dimensions
        head_activations = attention_weights.mean(dim=(-2, -1))  # (batch, num_heads)
        
        # Count active heads
        active_heads = (head_activations > threshold).float()
        
        # Calculate percentage
        cud = active_heads.mean().item()
        return cud
    else:
        raise ValueError(f"Unexpected attention_weights shape: {attention_weights.shape}")


def calculate_ape(
    attention_weights: torch.Tensor,
    epsilon: float = 1e-10,
    layer_wise: bool = False,
) -> Union[float, torch.Tensor]:
    """
    Calculate Attention Process Entropy (APE).
    
    APE measures the entropy of attention distributions, indicating
    how "focused" or "distributed" the attention is.
    
    Lower APE = More focused attention = More structured flow
    
    Args:
        attention_weights: Tensor of shape (batch, num_heads, seq_len, seq_len)
                          or (num_layers, batch, num_heads, seq_len, seq_len)
        epsilon: Small value to avoid log(0)
        layer_wise: If True, return APE per layer. If False, return average.
        
    Returns:
        APE value(s) as float or tensor
    """
    if attention_weights.dim() == 5:
        # Multi-layer: (num_layers, batch, num_heads, seq_len, seq_len)
        num_layers, batch, num_heads, seq_len, _ = attention_weights.shape
        
        # Flatten for entropy calculation: (num_layers, batch, num_heads, seq_len * seq_len)
        flat_attn = attention_weights.reshape(num_layers, batch, num_heads, -1)
        
        # Add epsilon to avoid log(0)
        flat_attn = flat_attn + epsilon
        
        # Normalize to get probability distributions
        flat_attn = flat_attn / flat_attn.sum(dim=-1, keepdim=True)
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -(flat_attn * torch.log(flat_attn)).sum(dim=-1)  # (num_layers, batch, num_heads)
        
        # Average over batch and heads
        ape_per_layer = entropy.mean(dim=(-2, -1))  # (num_layers,)
        
        if layer_wise:
            return ape_per_layer
        else:
            return ape_per_layer.mean().item()
            
    elif attention_weights.dim() == 4:
        # Single layer: (batch, num_heads, seq_len, seq_len)
        batch, num_heads, seq_len, _ = attention_weights.shape
        
        # Flatten
        flat_attn = attention_weights.reshape(batch, num_heads, -1)
        
        # Add epsilon and normalize
        flat_attn = flat_attn + epsilon
        flat_attn = flat_attn / flat_attn.sum(dim=-1, keepdim=True)
        
        # Calculate entropy
        entropy = -(flat_attn * torch.log(flat_attn)).sum(dim=-1)  # (batch, num_heads)
        
        # Average
        ape = entropy.mean().item()
        return ape
    else:
        raise ValueError(f"Unexpected attention_weights shape: {attention_weights.shape}")


def calculate_metrics_for_segment(
    attention_weights: torch.Tensor,
    start_idx: int,
    end_idx: int,
    threshold: float = 1e-3,
) -> dict:
    """
    Calculate CUD and APE for a specific token segment.
    
    Args:
        attention_weights: Full attention tensor (num_layers, batch, num_heads, seq_len, seq_len)
        start_idx: Start token index (inclusive)
        end_idx: End token index (exclusive)
        threshold: Threshold for CUD calculation
        
    Returns:
        Dictionary with 'cud' and 'ape' values
    """
    # Slice attention weights for the segment
    # attention_weights shape: (num_layers, batch, num_heads, seq_len, seq_len)
    segment_attn = attention_weights[:, :, :, start_idx:end_idx, start_idx:end_idx]
    
    # Calculate metrics
    cud = calculate_cud(segment_attn, threshold=threshold, layer_wise=False)
    ape = calculate_ape(segment_attn, layer_wise=False)
    
    return {
        "cud": cud,
        "ape": ape,
        "num_tokens": end_idx - start_idx,
    }

