"""
Training configuration dataclass.

Each finetuning phase imports this and overrides fields as needed.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class TrainConfig:
    """Configuration for ARC-AGI finetuning."""
    
    # === Phase Configuration ===
    phase: int = 1  # Training phase (1=fresh, 2/3=from checkpoint)
    checkpoint: Optional[str] = None  # Path to checkpoint for phases 2+
    
    # === Model ===
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    # === Data ===
    data_dir: str = ""  # Required: directory containing training data shards
    eval_dir: Optional[str] = None  # Optional: evaluation data directory
    max_seq_length: int = 16384  # Skip samples longer than this (OOM protection)
    
    # === Training Hyperparameters ===
    # Batch size: samples per forward pass. Larger = more efficient but more VRAM.
    # With padding, all samples in batch are padded to longest sequence.
    # Recommended: 1-2 for single GPU, 2-4 with FSDP on 4+ GPUs
    batch_size: int = 1
    # Effective batch = batch_size × grad_accum_steps × num_gpus
    grad_accum_steps: int = 32  # Accumulate until 32 samples processed
    epochs: int = 1  # Number of training epochs
    max_steps: Optional[int] = None  # Max steps (overrides epochs if set)
    lr: float = 2e-5  # Peak learning rate
    warmup_steps: int = 100  # Warmup steps
    weight_decay: float = 0.01  # Weight decay
    max_grad_norm: float = 1.0  # Max gradient norm for clipping
    
    # === Logging and Saving ===
    output_dir: str = "./checkpoints"  # Output directory for checkpoints
    save_steps: int = 10000  # Save checkpoint every N steps
    log_steps: int = 100  # Log metrics every N steps
    eval_steps: int = 1000  # Evaluate every N steps (quick sampled eval)
    eval_samples: Optional[int] = 100  # Samples per eval (None = full eval set)
    save_optimizer_state: bool = False  # Model-only checkpoints (~15GB vs ~76GB)
    
    # === Weights & Biases ===
    use_wandb: bool = True  # Enable W&B logging
    wandb_project: str = "arc_agi"  # W&B project name
    wandb_run_name: Optional[str] = None  # Run name (auto-generated if None)
    wandb_entity: Optional[str] = None  # W&B team/entity (None = personal)
    
    # === Hardware ===
    num_workers: int = 4  # DataLoader workers
    dtype: str = "bfloat16"  # Training dtype: float32, float16, bfloat16
    use_amp: bool = True  # Use automatic mixed precision
    
    # === Memory Optimization ===
    use_8bit_adam: bool = True  # Use bitsandbytes 8-bit AdamW
    gradient_checkpointing: bool = True  # Trade compute for memory
    
    # === Performance Optimization ===
    use_torch_compile: bool = True  # Use torch.compile for faster training (1.5-2x speedup)
    compile_mode: str = "default"  # "default", "reduce-overhead", or "max-autotune"
    
    # === Distributed Training (FSDP) ===
    distributed: bool = False  # Enable distributed training
    sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    cpu_offload: bool = False  # Offload params to CPU (slower but saves GPU memory)
    
    # === Loss Configuration ===
    fim_ratio: float = 0.5  # Fraction of samples to use FIM format (0=all NTP, 1=all FIM)
    loss_on_output_only: bool = False  # If True, only compute loss on output grid pixels (not metadata)
    
    @property
    def effective_batch_size(self) -> int:
        """Effective batch size accounting for gradient accumulation."""
        return self.batch_size * self.grad_accum_steps
    
    @property
    def torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.dtype]
    
    def validate(self) -> None:
        """Validate configuration."""
        if not self.data_dir:
            raise ValueError("data_dir is required")
        
        if self.phase > 1 and self.checkpoint is None:
            raise ValueError(f"Phase {self.phase} requires a checkpoint path")
        
        if self.phase not in [1, 2, 3]:
            raise ValueError(f"Phase must be 1, 2, or 3, got {self.phase}")


# === Phase-specific default configs ===

@dataclass
class Phase1Config(TrainConfig):
    """Phase 1: Initial finetuning on synthetic data."""
    phase: int = 1
    epochs: int = 3
    lr: float = 2e-5
    fim_ratio=.5


@dataclass
class Phase2Config(TrainConfig):
    """Phase 2: Continue training on curated data."""
    phase: int = 2
    epochs: int = 2
    lr: float = 1e-5  # Lower LR for phase 2
    loss_on_output_only: bool = True  # Focus loss on output grid pixels


@dataclass  
class Phase3Config(TrainConfig):
    """Phase 3: Final refinement on hard examples."""
    phase: int = 3
    epochs: int = 1
    lr: float = 5e-6  # Even lower LR for final phase
    loss_on_output_only: bool = True  # Focus loss on output grid pixels

