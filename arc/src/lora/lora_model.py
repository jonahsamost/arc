"""
LoRA (Low-Rank Adaptation) implementation for Qwen models.

Provides:
- LoRALinear: Low-rank adapter layer that wraps existing Linear layers
- apply_lora_to_model(): Inject LoRA into specified attention projections
- Utilities for saving/loading LoRA weights separately from base model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
import math

from src.lora.config import LoRAConfig


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation layer that wraps a frozen Linear layer.
    
    Computes: output = base_layer(x) + (x @ A @ B) * scaling
    
    Where:
    - base_layer: Original frozen Linear layer
    - A: [in_features, r] - learned down-projection
    - B: [r, out_features] - learned up-projection  
    - scaling: alpha / r
    
    This allows fine-tuning with far fewer parameters than full fine-tuning.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        init_b_zero: bool = True,
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(in_features, r))
        self.lora_B = nn.Parameter(torch.empty(r, out_features))
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        self._init_weights(init_b_zero)
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def _init_weights(self, init_b_zero: bool = True):
        """
        Initialize LoRA weights.
        
        Standard LoRA init:
        - A: Kaiming uniform (same as Linear default)
        - B: Zero (so initial output matches base layer exactly)
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if init_b_zero:
            nn.init.zeros_(self.lora_B)
        else:
            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: base output + LoRA delta.
        """
        # Base layer output (frozen)
        base_out = self.base_layer(x)
        
        # LoRA path: x @ A @ B * scaling
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        
        return base_out + lora_out
    
    def merge_weights(self) -> None:
        """
        Merge LoRA weights into base layer (for inference efficiency).
        After merging, forward() just calls base_layer().
        """
        with torch.no_grad():
            # W_new = W_base + A @ B * scaling
            delta = (self.lora_A @ self.lora_B * self.scaling).T
            self.base_layer.weight.add_(delta)
    
    def extra_repr(self) -> str:
        return f"in={self.base_layer.in_features}, out={self.base_layer.out_features}, r={self.r}, alpha={self.alpha}"


def apply_lora_to_model(
    model,
    config: LoRAConfig,
) -> List[str]:
    """
    Apply LoRA adapters to specified modules in the model.
    
    Args:
        model: The Qwen model (with 2D RoPE surgery already applied)
        config: LoRA configuration
    
    Returns:
        List of module names that were wrapped with LoRA
    """
    target_modules = set(config.target_modules)
    wrapped_modules = []
    
    # Find and wrap target modules in attention layers
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        
        for name in target_modules:
            if hasattr(attn, name):
                base_layer = getattr(attn, name)
                if isinstance(base_layer, nn.Linear):
                    # Wrap with LoRA
                    lora_layer = LoRALinear(
                        base_layer=base_layer,
                        r=config.r,
                        alpha=config.alpha,
                        dropout=config.dropout,
                        init_b_zero=config.init_b_zero,
                    )
                    setattr(attn, name, lora_layer)
                    wrapped_modules.append(f"layers.{layer_idx}.self_attn.{name}")
    
    print(f"Applied LoRA (r={config.r}, alpha={config.alpha}) to {len(wrapped_modules)} modules")
    return wrapped_modules


def get_lora_params(model) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from the model.
    
    Returns list of parameters for optimizer.
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_params.append(module.lora_A)
            lora_params.append(module.lora_B)
    return lora_params


def get_lora_state_dict(model) -> Dict[str, torch.Tensor]:
    """
    Extract LoRA weights as a state dict.
    
    Returns dict mapping module paths to LoRA A and B tensors.
    """
    state_dict = {}
    
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if hasattr(attn, name):
                module = getattr(attn, name)
                if isinstance(module, LoRALinear):
                    prefix = f"layers.{layer_idx}.self_attn.{name}"
                    state_dict[f"{prefix}.lora_A"] = module.lora_A.data.clone()
                    state_dict[f"{prefix}.lora_B"] = module.lora_B.data.clone()
    
    return state_dict


def load_lora_state_dict(model, state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Load LoRA weights from a state dict into the model.
    """
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if hasattr(attn, name):
                module = getattr(attn, name)
                if isinstance(module, LoRALinear):
                    prefix = f"layers.{layer_idx}.self_attn.{name}"
                    a_key = f"{prefix}.lora_A"
                    b_key = f"{prefix}.lora_B"
                    
                    if a_key in state_dict:
                        module.lora_A.data.copy_(state_dict[a_key])
                    if b_key in state_dict:
                        module.lora_B.data.copy_(state_dict[b_key])


def save_lora(model, path: str) -> None:
    """
    Save LoRA weights to a file.
    """
    state_dict = get_lora_state_dict(model)
    torch.save(state_dict, path)
    
    # Report size
    size_mb = sum(t.numel() * t.element_size() for t in state_dict.values()) / (1024 * 1024)
    print(f"Saved LoRA weights to {path} ({size_mb:.2f} MB, {len(state_dict)} tensors)")


def load_lora(model, path: str) -> None:
    """
    Load LoRA weights from a file.
    """
    state_dict = torch.load(path, map_location="cpu")
    load_lora_state_dict(model, state_dict)
    print(f"Loaded LoRA weights from {path} ({len(state_dict)} tensors)")


def freeze_base_model(model) -> int:
    """
    Freeze all parameters in the base model (non-LoRA).
    
    Returns number of frozen parameters.
    """
    frozen_count = 0
    
    for name, param in model.named_parameters():
        # Skip LoRA parameters
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            frozen_count += param.numel()
    
    # Count trainable
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Frozen {frozen_count:,} parameters, {trainable_count:,} trainable (LoRA)")
    return frozen_count


def count_lora_params(model) -> int:
    """Count total number of LoRA parameters."""
    return sum(p.numel() for p in get_lora_params(model))
