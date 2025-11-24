"""
Activation Energy (AE) metric calculation.

AE measures the L2 norm of hidden states, indicating the overall
activation magnitude in the model.
"""

import torch
from typing import Union


def calculate_activation_energy(
    hidden_states: torch.Tensor,
    layer_wise: bool = False,
) -> Union[float, torch.Tensor]:
    """
    Calculate Activation Energy (AE) - L2 norm of hidden states.
    
    AE measures the overall magnitude of activations, which may be
    lower or more peaked during specific reasoning steps.
    
    Args:
        hidden_states: Tensor of shape (batch, seq_len, hidden_size)
                     or (num_layers, batch, seq_len, hidden_size)
        layer_wise: If True, return AE per layer. If False, return average.
        
    Returns:
        AE value(s) as float or tensor
    """
    if hidden_states.dim() == 4:
        # Multi-layer: (num_layers, batch, seq_len, hidden_size)
        # Calculate L2 norm for each token position
        l2_norms = torch.norm(hidden_states, p=2, dim=-1)  # (num_layers, batch, seq_len)
        
        # Average over sequence and batch
        ae_per_layer = l2_norms.mean(dim=(-2, -1))  # (num_layers,)
        
        if layer_wise:
            return ae_per_layer
        else:
            return ae_per_layer.mean().item()
            
    elif hidden_states.dim() == 3:
        # Single layer: (batch, seq_len, hidden_size)
        # Calculate L2 norm for each token position
        l2_norms = torch.norm(hidden_states, p=2, dim=-1)  # (batch, seq_len)
        
        # Average over sequence and batch
        ae = l2_norms.mean().item()
        return ae
    else:
        raise ValueError(f"Unexpected hidden_states shape: {hidden_states.shape}")


def calculate_ae_for_segment(
    hidden_states: torch.Tensor,
    start_idx: int,
    end_idx: int,
) -> dict:
    """
    Calculate Activation Energy for a specific token segment.
    
    Args:
        hidden_states: Full hidden states tensor (num_layers, batch, seq_len, hidden_size)
        start_idx: Start token index (inclusive)
        end_idx: End token index (exclusive)
        
    Returns:
        Dictionary with 'ae' value and 'num_tokens'
    """
    # Slice hidden states for the segment
    segment_states = hidden_states[:, :, start_idx:end_idx, :]
    
    # Calculate AE
    ae = calculate_activation_energy(segment_states, layer_wise=False)
    
    return {
        "ae": ae,
        "num_tokens": end_idx - start_idx,
    }

