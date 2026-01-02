"""
Qwen 2D RoPE Model Loader

This module handles:
1. Loading Qwen-2.5-Coder with proper configuration
2. Adding special tokens and resizing embeddings  
3. RoPE surgery: modifying attention to support 2D positional encoding
4. Checkpoint loading/saving with phase awareness

Design (Pure Mode Switching):
- Text tokens: 100% 1D RoPE (sequential position encoding)
- Grid tokens: 100% 2D RoPE (spatial position encoding)
- grid_mode mask determines which encoding each token uses
- No head splitting - each token uses full head_dim for its position type
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)
import math


class Qwen2DRotaryEmbedding(nn.Module):
    """
    Extended RoPE with pure mode switching between 1D and 2D.
    
    For head_dim=128:
    - 1D mode (text tokens): All 128 dims encode sequential position
    - 2D mode (grid tokens): 64 dims for y-axis, 64 dims for x-axis
    
    No head splitting - each token uses its full representation for position.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: int = 10000,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim  # head_dim, e.g. 128
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # 1D RoPE frequencies (for text tokens)
        # inv_freq has dim/2 = 64 values -> produces 128-dim cos/sin
        inv_freq_1d = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq_1d", inv_freq_1d, persistent=False)
        
        # 2D RoPE frequencies (for grid tokens)
        # Split head_dim in half: 64 for y-axis, 64 for x-axis
        # Each axis needs 32 base frequencies -> produces 64-dim cos/sin each
        axis_dim = dim // 2  # 64
        inv_freq_2d = 1.0 / (
            self.base ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim)
        )
        self.register_buffer("inv_freq_2d", inv_freq_2d, persistent=False)  # 32 values
    
    def _compute_1d_freqs(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 1D cos/sin for sequential positions."""
        # position_ids: [batch, seq_len]
        inv_freq = self.inv_freq_1d.to(position_ids.device)
        
        # [batch, seq_len, dim/2] = [batch, seq, 64]
        freqs = torch.einsum("bi,j->bij", position_ids.float(), inv_freq)
        # [batch, seq_len, dim] = [batch, seq, 128]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        return emb.cos(), emb.sin()
    
    def _compute_2d_freqs(
        self, 
        pos_y: torch.Tensor, 
        pos_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 2D cos/sin for spatial positions.
        
        Output shape: [batch, seq, 128] (full head_dim)
        - dims [0:64]:   y-axis rotation
        - dims [64:128]: x-axis rotation
        """
        # pos_y, pos_x: [batch, seq_len]
        inv_freq = self.inv_freq_2d.to(pos_y.device)  # [32]
        
        # Compute base frequencies for each axis
        # [batch, seq_len, 32]
        freqs_y = torch.einsum("bi,j->bij", pos_y.float(), inv_freq)
        freqs_x = torch.einsum("bi,j->bij", pos_x.float(), inv_freq)
        
        # Expand for rotate_half compatibility: [f0, f0, f1, f1, ...]
        # [batch, seq_len, 64] each
        emb_y = torch.cat([freqs_y, freqs_y], dim=-1)
        emb_x = torch.cat([freqs_x, freqs_x], dim=-1)
        
        # Concatenate y and x portions
        # [batch, seq_len, 128] - full head_dim
        emb = torch.cat([emb_y, emb_x], dim=-1)
        
        return emb.cos(), emb.sin()
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        pos_2d: Optional[torch.Tensor] = None,
        grid_mode: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute rotary embeddings for both modes.
        
        Args:
            x: Input tensor (used only for device)
            position_ids: [batch, seq_len] 1D positions
            pos_2d: [batch, seq_len, 2] optional (y, x) coordinates
            grid_mode: [batch, seq_len] optional mask (0=text, 1=grid)
        
        Returns:
            cos_1d, sin_1d: [batch, seq, head_dim] for 1D mode
            cos_2d, sin_2d: [batch, seq, head_dim] for 2D mode (None if no 2D info)
            grid_mode: passed through for apply function
        """
        cos_1d, sin_1d = self._compute_1d_freqs(position_ids)
        
        if pos_2d is not None and grid_mode is not None:
            pos_y = pos_2d[..., 0]  # [batch, seq_len]
            pos_x = pos_2d[..., 1]  # [batch, seq_len]
            cos_2d, sin_2d = self._compute_2d_freqs(pos_y, pos_x)
            return cos_1d, sin_1d, cos_2d, sin_2d, grid_mode
        
        return cos_1d, sin_1d, None, None, None


def apply_rotary_pos_emb_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_1d: torch.Tensor,
    sin_1d: torch.Tensor,
    cos_2d: Optional[torch.Tensor],
    sin_2d: Optional[torch.Tensor],
    grid_mode: Optional[torch.Tensor],
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings with pure mode switching.
    
    Strategy:
    - Text tokens (grid_mode=0): Apply 1D RoPE to full head_dim
    - Grid tokens (grid_mode=1): Apply 2D RoPE to full head_dim
    
    Args:
        q, k: [batch, heads, seq, head_dim]
        cos_1d, sin_1d: [batch, seq, head_dim] 1D frequencies
        cos_2d, sin_2d: [batch, seq, head_dim] 2D frequencies (optional)
        grid_mode: [batch, seq] mask - 0 for text, 1 for grid (optional)
        unsqueeze_dim: Dimension to add for head broadcasting
    """
    # Cast cos/sin to match q/k dtype (they're computed in float32 for precision)
    dtype = q.dtype
    cos_1d = cos_1d.unsqueeze(unsqueeze_dim).to(dtype)
    sin_1d = sin_1d.unsqueeze(unsqueeze_dim).to(dtype)
    
    # Apply 1D rotation (baseline for all tokens)
    q_1d = (q * cos_1d) + (rotate_half(q) * sin_1d)
    k_1d = (k * cos_1d) + (rotate_half(k) * sin_1d)
    
    # If no 2D info, just return 1D result
    if cos_2d is None or sin_2d is None or grid_mode is None:
        return q_1d, k_1d
    
    # Unsqueeze and cast 2D frequencies
    cos_2d = cos_2d.unsqueeze(unsqueeze_dim).to(dtype)
    sin_2d = sin_2d.unsqueeze(unsqueeze_dim).to(dtype)
    
    # Apply 2D rotation
    q_2d = (q * cos_2d) + (rotate_half(q) * sin_2d)
    k_2d = (k * cos_2d) + (rotate_half(k) * sin_2d)
    
    # Select based on grid_mode: text tokens use 1D, grid tokens use 2D
    # grid_mode: [batch, seq] -> [batch, 1, seq, 1] for broadcasting
    mask = grid_mode.unsqueeze(1).unsqueeze(-1).float()
    
    q_embed = mask * q_2d + (1 - mask) * q_1d
    k_embed = mask * k_2d + (1 - mask) * k_1d
    
    return q_embed, k_embed


class Qwen2DAttention(nn.Module):
    """
    Wrapper that replaces Qwen2Attention's forward to use 2D RoPE.
    
    This is a surgical replacement - we keep all the original weights
    but change how rotary embeddings are applied.
    """
    
    def __init__(self, original_attention: Qwen2Attention, config):
        super().__init__()
        # Copy all attributes from original
        self.config = config
        self.layer_idx = original_attention.layer_idx
        self.hidden_size = original_attention.config.hidden_size
        self.head_dim = original_attention.head_dim
        self.num_heads = original_attention.config.num_attention_heads
        self.num_key_value_heads = original_attention.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # GQA grouping
        self.max_position_embeddings = original_attention.config.max_position_embeddings
        self.rope_theta = original_attention.config.rope_theta
        self.is_causal = original_attention.is_causal
        self.attention_dropout = original_attention.attention_dropout
        
        # Keep original projection layers (these have the trained weights)
        self.q_proj = original_attention.q_proj
        self.k_proj = original_attention.k_proj
        self.v_proj = original_attention.v_proj
        self.o_proj = original_attention.o_proj
        
        # Replace rotary embedding with 2D-capable version
        self.rotary_emb = Qwen2DRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        # New 2D arguments (can also be accessed via self._pos_2d, self._grid_mode)
        pos_2d: Optional[torch.Tensor] = None,
        grid_mode: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        # Check for stored 2D position info (set by training loop)
        if pos_2d is None:
            pos_2d = getattr(self, '_pos_2d', None)
        if grid_mode is None:
            grid_mode = getattr(self, '_grid_mode', None)
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Compute rotary embeddings (1D and optionally 2D)
        cos_1d, sin_1d, cos_2d, sin_2d, grid_mode_out = self.rotary_emb(
            value_states, 
            position_ids,
            pos_2d=pos_2d,
            grid_mode=grid_mode,
        )
        
        # Apply hybrid 1D/2D rotary embeddings
        query_states, key_states = apply_rotary_pos_emb_2d(
            query_states, key_states,
            cos_1d, sin_1d,
            cos_2d, sin_2d,
            grid_mode_out,
        )
        
        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Repeat KV for grouped-query attention
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Use PyTorch's scaled_dot_product_attention (Flash Attention when available)
        dropout_p = self.attention_dropout if self.training else 0.0
        
        # Prepare attention mask for SDPA
        # attention_mask from HuggingFace is [batch, seq] with 1=real, 0=padding
        # SDPA needs [batch, 1, seq, seq] with 0=attend, -inf=mask
        sdpa_mask = None
        use_causal = self.is_causal
        
        if attention_mask is not None and attention_mask.dim() == 2:
            # Convert [batch, seq] padding mask to [batch, 1, 1, seq] for key masking
            # Then combine with causal mask
            bsz, seq_len = attention_mask.shape
            
            # Create causal mask: [1, 1, seq, seq]
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=attention_mask.device, dtype=query_states.dtype),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)
            
            # Create padding mask: [batch, 1, 1, seq] - mask where attention_mask is 0
            padding_mask = attention_mask[:, None, None, :].to(query_states.dtype)
            padding_mask = (1.0 - padding_mask) * float("-inf")
            
            # Combine: causal + padding (both use -inf for masked positions)
            # Broadcasting: [1, 1, seq, seq] + [batch, 1, 1, seq] -> [batch, 1, seq, seq]
            sdpa_mask = causal_mask + padding_mask
            use_causal = False  # We're providing our own causal mask
        
        attn_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states, 
            value_states,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
            is_causal=use_causal,
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Reshape: [bsz, seq_len, num_heads, head_dim] -> [bsz, seq_len, num_heads * head_dim]
        # Note: num_heads * head_dim may differ from hidden_size (e.g., in some model configs)
        # The o_proj layer will project from num_heads * head_dim back to hidden_size
        actual_seq_len = attn_output.size(1)
        attn_output = attn_output.reshape(bsz, actual_seq_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        
        # Qwen2DecoderLayer expects (attn_output, attn_weights) - only 2 values
        return attn_output, None


def apply_rope_surgery(model) -> None:
    """
    Replace all attention layers with 2D-capable versions.
    
    This modifies the model in-place.
    """
    config = model.config
    
    for layer_idx, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn
        new_attn = Qwen2DAttention(old_attn, config)
        layer.self_attn = new_attn
    
    print(f"Applied 2D RoPE surgery to {len(model.model.layers)} attention layers")


def load_qwen_2d(
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    tokenizer_2d = None,
    checkpoint_path: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> Tuple[Any, Any]:
    """
    Load Qwen model with 2D RoPE modifications.
    
    Args:
        model_name: HuggingFace model name
        tokenizer_2d: Arc2DTokenizer instance (for adding special tokens)
        checkpoint_path: Path to load finetuned weights from (for phases 2+)
        dtype: Model dtype
        device: Target device
    
    Returns:
        (model, tokenizer_2d)
    """
    print(f"Loading Qwen from {model_name}...")
    
    # Load base model (CPU first for embedding resize)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=None,  # Load to CPU first
        trust_remote_code=True,
    )
    
    # Add special tokens and resize embeddings
    if tokenizer_2d is not None:
        tokenizer, model = tokenizer_2d.add_special_tokens_to_model(model)
    
    # Apply 2D RoPE surgery
    apply_rope_surgery(model)
    
    # Load checkpoint if provided (phase 2+)
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    
    # Move to device
    model = model.to(device)
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, tokenizer_2d


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    loss: float,
    path: str,
    phase: int = 1,
    save_optimizer_state: bool = True,
):
    """
    Save training checkpoint.
    
    Args:
        save_optimizer_state: If False, only save model weights (~15GB vs ~76GB).
                             Set False for final phase checkpoints to save disk.
    
    Saves:
    - Model state dict
    - Optimizer state dict (if save_optimizer_state=True)
    - Scheduler state dict (if save_optimizer_state=True)
    - Training metadata
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if save_optimizer_state else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler and save_optimizer_state else None,
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "phase": phase,
    }
    torch.save(checkpoint, path)
    
    import os
    size_gb = os.path.getsize(path) / (1024**3)
    print(f"Checkpoint saved to {path} ({size_gb:.1f} GB)")


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Returns metadata dict with epoch, step, loss, phase.
    """
    checkpoint = torch.load(path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", float("inf")),
        "phase": checkpoint.get("phase", 1),
    }

