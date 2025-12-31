"""
Loss functions for ARC-AGI finetuning.

Supports:
- Next Token Prediction (NTP) loss
- Fill-In-Middle (FIM) loss (TODO)
- Combined loss with weighting
"""

import torch
import torch.nn as nn
from torch.amp import autocast
from typing import Dict, Any, Optional


def compute_ntp_loss(
    model,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    loss_on_output_only: bool = False,
) -> torch.Tensor:
    """
    Compute Next Token Prediction loss.
    
    Handles 2D position encoding by storing pos_2d and grid_mode
    on attention layers before forward pass.
    
    Args:
        model: The Qwen model with 2D RoPE surgery applied
        batch: Collated batch with input_ids, labels, attention_mask, pos_1d, pos_2d, grid_mode, output_mask
        device: Target device
        use_amp: Whether to use automatic mixed precision
        amp_dtype: Dtype for AMP (bfloat16 or float16)
        loss_on_output_only: If True, only compute loss on output grid pixels (output_mask=1)
    
    Returns:
        Scalar loss tensor
    """
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pos_1d = batch["pos_1d"].to(device)
    pos_2d = batch["pos_2d"].to(device)
    grid_mode = batch["grid_mode"].to(device)
    
    # Store 2D position info on attention layers
    # These are accessed during forward pass by Qwen2DAttention
    _set_2d_positions(model, pos_2d, grid_mode)
    
    try:
        # Forward pass with optional AMP
        device_type = "cuda" if device.type == "cuda" else "cpu"
        with autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=pos_1d,
                use_cache=False,
            )
            
            logits = outputs.logits
            
            # Shift for next token prediction: predict token[i+1] from token[i]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # If loss_on_output_only, mask out non-output tokens
            if loss_on_output_only and "output_mask" in batch:
                output_mask = batch["output_mask"].to(device)
                # Shift output_mask to align with shifted labels
                shift_output_mask = output_mask[..., 1:].contiguous()
                # Set labels to -100 where output_mask is 0 (non-output tokens)
                shift_labels = shift_labels.clone()
                shift_labels[shift_output_mask == 0] = -100
            
            # Flatten for cross entropy
            vocab_size = shift_logits.size(-1)
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # Cross entropy loss (automatically ignores -100 labels)
            # Note: loss is computed in float32 for stability even with AMP
            loss = nn.functional.cross_entropy(
                shift_logits.float(), 
                shift_labels, 
                ignore_index=-100,
            )
        
        return loss
    
    finally:
        # Clean up references (not needed for backprop, just memory hygiene)
        _clear_2d_positions(model)


def _set_2d_positions(model, pos_2d: torch.Tensor, grid_mode: torch.Tensor) -> None:
    """Store 2D position tensors on attention layers for forward pass access."""
    for layer in model.model.layers:
        layer.self_attn._pos_2d = pos_2d
        layer.self_attn._grid_mode = grid_mode


def _clear_2d_positions(model) -> None:
    """Clear 2D position tensors from attention layers."""
    for layer in model.model.layers:
        layer.self_attn._pos_2d = None
        layer.self_attn._grid_mode = None


def compute_fim_loss(
    model,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute Fill-In-Middle loss.
    
    FIM format: <|fim_prefix|> [context] <|fim_suffix|> [suffix] <|fim_middle|> [middle]
    
    The model learns to predict the masked middle section given prefix and suffix context.
    For ARC grids, this helps learn spatial patterns (fill in a region given surrounding pixels).
    
    TODO: Implement FIM sample generation in tokenizer first
    """
    raise NotImplementedError("FIM loss requires FIM sample generation in tokenizer")


def compute_combined_loss(
    model,
    ntp_batch: Dict[str, torch.Tensor],
    fim_batch: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
    fim_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Compute combined NTP + FIM loss.
    
    Args:
        model: The model
        ntp_batch: Batch for next token prediction
        fim_batch: Batch for fill-in-middle (optional)
        device: Target device
        fim_weight: Weight for FIM loss (NTP gets 1 - fim_weight)
    
    Returns:
        Dict with 'loss', 'ntp_loss', and optionally 'fim_loss'
    """
    ntp_loss = compute_ntp_loss(model, ntp_batch, device)
    
    if fim_batch is not None:
        fim_loss = compute_fim_loss(model, fim_batch, device)
        total_loss = (1 - fim_weight) * ntp_loss + fim_weight * fim_loss
        return {
            "loss": total_loss,
            "ntp_loss": ntp_loss,
            "fim_loss": fim_loss,
        }
    
    return {
        "loss": ntp_loss,
        "ntp_loss": ntp_loss,
    }

