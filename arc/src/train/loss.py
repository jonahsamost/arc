"""
Loss functions for ARC-AGI finetuning.

Supports:
- Next Token Prediction (NTP) loss
- Fill-In-Middle (FIM) loss (TODO)
- Combined loss with weighting
"""

import os
import torch
import torch.nn as nn
from torch.amp import autocast
from typing import Dict, Any, Optional

# Debug flag - set to True to see detailed forward pass logging
_DEBUG_FORWARD = False


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
    rank = int(os.environ.get("RANK", 0))
    
    if _DEBUG_FORWARD and rank == 0:
        print(f"[Loss] Moving tensors to device...", flush=True)
    
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pos_1d = batch["pos_1d"].to(device)
    pos_2d = batch["pos_2d"].to(device)
    grid_mode = batch["grid_mode"].to(device)
    
    if _DEBUG_FORWARD and rank == 0:
        print(f"[Loss] Tensors on device. input_ids shape: {input_ids.shape}", flush=True)
    
    # Store 2D position info on attention layers
    # These are accessed during forward pass by Qwen2DAttention
    if _DEBUG_FORWARD and rank == 0:
        print(f"[Loss] Setting 2D positions...", flush=True)
    _set_2d_positions(model, pos_2d, grid_mode)
    
    if _DEBUG_FORWARD and rank == 0:
        print(f"[Loss] 2D positions set. Starting model forward...", flush=True)
    
    # Forward pass with optional AMP
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
        if _DEBUG_FORWARD and rank == 0:
            print(f"[Loss] Inside autocast, calling model()...", flush=True)
        print(f'shapes: input_ids: {input_ids.shape}, attn: {attention_mask.shape}, pos: {pos_1d.shape}')
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=pos_1d,
            use_cache=False,
        )
        if _DEBUG_FORWARD and rank == 0:
            print(f"[Loss] model() returned!", flush=True)
        
        # Extract logits and immediately free model outputs to save memory
        logits = outputs.logits
        del outputs
        
        # Shift for next token prediction: predict token[i+1] from token[i]
        # Note: We slice logits in-place to avoid duplicating the full tensor
        shift_logits = logits[..., :-1, :]
        del logits  # Free reference to original logits
        shift_logits = shift_logits.contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # If loss_on_output_only, mask out non-output tokens
        if loss_on_output_only and "output_mask" in batch:
            output_mask = batch["output_mask"].to(device)
            # Shift output_mask to align with shifted labels
            shift_output_mask = output_mask[..., 1:].contiguous()
            # Set labels to -100 where output_mask is 0 (non-output tokens)
            shift_labels = shift_labels.clone()
            shift_labels[shift_output_mask == 0] = -100
        
        # Use chunked cross-entropy for long sequences to avoid OOM
        # For sequences > 4096 tokens, compute loss in chunks
        seq_len = shift_logits.size(1)
        chunk_size = 2048  # Reduced from 4096 for more aggressive chunking
        
        if seq_len > chunk_size:
            # Chunked cross-entropy: avoids materializing full [batch, seq, vocab] at once
            rank = int(os.environ.get("RANK", 0))
            # print(f"[Rank {rank}] [Loss] Using chunked CE: seq_len={seq_len}, chunk_size={chunk_size}, batch={shift_logits.size(0)}", flush=True)
            loss = _chunked_cross_entropy(shift_logits, shift_labels, chunk_size)
            # print(f"[Rank {rank}] [Loss] Chunked CE returned, loss={loss.item():.4f}", flush=True)
        else:
            # Standard cross-entropy for shorter sequences
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
    
    rank = int(os.environ.get("RANK", 0))
    # print(f"[Rank {rank}] [Loss] compute_ntp_loss returning, loss={loss.item():.4f}", flush=True)
    return loss


def _chunked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Compute cross-entropy loss in chunks to avoid OOM on long sequences.
    
    Instead of materializing full [batch, seq, vocab] tensor for loss computation,
    we process chunks of the sequence dimension at a time.
    
    Args:
        logits: [batch, seq_len, vocab_size] - model output logits
        labels: [batch, seq_len] - target token IDs (-100 = ignore)
        chunk_size: Number of sequence positions to process at once
    
    Returns:
        Scalar loss tensor (mean over all non-ignored positions)
    """
    rank = int(os.environ.get("RANK", 0))
    batch_size, seq_len, vocab_size = logits.shape
    
    # print(f'[Rank {rank}] _chunked_cross_entropy: batch={batch_size}, seq_len={seq_len}, vocab={vocab_size}, chunk_size={chunk_size}', flush=True)
    # print(f'[Rank {rank}] Processing {((seq_len + chunk_size - 1) // chunk_size)} chunks...', flush=True)
    
    total_loss = 0.0
    total_tokens = 0
    
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    for chunk_idx, start_idx in enumerate(range(0, seq_len, chunk_size)):
        end_idx = min(start_idx + chunk_size, seq_len)
        # print(f'[Rank {rank}] Chunk {chunk_idx+1}/{num_chunks}: [{start_idx}:{end_idx}]', flush=True)
        
        # Get chunk of logits and labels
        # print(f'[Rank {rank}] Slicing logits and labels...', flush=True)
        chunk_logits = logits[:, start_idx:end_idx, :].contiguous()
        chunk_labels = labels[:, start_idx:end_idx].contiguous()
        # print(f'[Rank {rank}] Slicing done', flush=True)
        
        # Flatten chunk
        # print(f'[Rank {rank}] Flattening chunk...', flush=True)
        chunk_logits = chunk_logits.view(-1, vocab_size)
        chunk_labels = chunk_labels.view(-1)
        # print(f'[Rank {rank}] Flattening done', flush=True)
        
        # Count valid tokens in this chunk (not -100)
        # print(f'[Rank {rank}] Counting valid tokens...', flush=True)
        valid_mask = chunk_labels != -100
        num_valid = valid_mask.sum().item()
        # print(f'[Rank {rank}] Valid tokens: {num_valid}', flush=True)
        
        if num_valid > 0:
            # Compute loss for this chunk (reduction='sum' to accumulate properly)
            # print(f'[Rank {rank}] Computing cross_entropy for chunk {chunk_idx+1}...', flush=True)
            chunk_loss = nn.functional.cross_entropy(
                chunk_logits.float(),
                chunk_labels,
                ignore_index=-100,
                reduction='sum',
            )
            # print(f'[Rank {rank}] Chunk {chunk_idx+1} loss computed: {chunk_loss.item():.4f}', flush=True)
            total_loss = total_loss + chunk_loss
            total_tokens += num_valid
        # else:
            # print(f'[Rank {rank}] Chunk {chunk_idx+1} has no valid tokens, skipping', flush=True)
    
    # Return mean loss over all valid tokens
    # print(f'[Rank {rank}] _chunked_cross_entropy: total_tokens={total_tokens}, final_loss={total_loss.item() / total_tokens if total_tokens > 0 else 0:.4f}', flush=True)
    if total_tokens > 0:
        return total_loss / total_tokens
    else:
        # Edge case: no valid tokens (shouldn't happen in practice)
        # print(f'[Rank {rank}] WARNING: No valid tokens in chunked CE!', flush=True)
        return torch.tensor(0.0, device=logits.device, requires_grad=True)


def _get_base_model(model):
    """Get the underlying model, handling FSDP wrapping."""
    # FSDP wraps modules - access via _fsdp_wrapped_module or module attribute
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    
    # Unwrap FSDP if present
    if isinstance(model, FSDP):
        # For FSDP, access the underlying module
        base = model._fsdp_wrapped_module if hasattr(model, '_fsdp_wrapped_module') else model.module
    else:
        base = model
    
    return base


def _set_2d_positions(model, pos_2d: torch.Tensor, grid_mode: torch.Tensor) -> None:
    """Store 2D position tensors on attention layers for forward pass access."""
    rank = int(os.environ.get("RANK", 0))
    base_model = _get_base_model(model)
    
    if _DEBUG_FORWARD and rank == 0:
        print(f"[Loss] _set_2d_positions: base_model type = {type(base_model).__name__}", flush=True)
    
    # Access layers - handle both wrapped and unwrapped cases
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        layers = base_model.model.layers
        if _DEBUG_FORWARD and rank == 0:
            print(f"[Loss] Found {len(layers)} layers via model.model.layers", flush=True)
    else:
        # Try to find layers by walking the module tree
        layers = []
        for name, module in base_model.named_modules():
            if name.endswith('.self_attn') or (hasattr(module, 'self_attn')):
                if hasattr(module, 'self_attn'):
                    layers.append(module)
        if _DEBUG_FORWARD and rank == 0:
            print(f"[Loss] Found {len(layers)} layers via named_modules walk", flush=True)
    
    count = 0
    for layer in layers:
        attn = layer.self_attn if hasattr(layer, 'self_attn') else layer
        # Handle FSDP-wrapped attention
        if hasattr(attn, '_fsdp_wrapped_module'):
            attn = attn._fsdp_wrapped_module
        attn._pos_2d = pos_2d
        attn._grid_mode = grid_mode
        count += 1
    
    if _DEBUG_FORWARD and rank == 0:
        print(f"[Loss] Set 2D positions on {count} attention modules", flush=True)


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

