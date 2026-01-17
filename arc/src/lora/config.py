"""
Configuration dataclasses for LoRA and Test-Time Training.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    
    # LoRA rank - lower = fewer params, higher = more expressive
    r: int = 4
    
    # Scaling factor: output = base(x) + (x @ A @ B) * (alpha / r)
    alpha: float = 1.0
    
    # Which modules to apply LoRA to
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj")
    
    # Dropout on LoRA path (0 = disabled)
    dropout: float = 0.0
    
    # Initialize B matrix to zero (standard LoRA init)
    init_b_zero: bool = True
    
    @property
    def scaling(self) -> float:
        """Compute LoRA scaling factor."""
        return self.alpha / self.r


@dataclass
class TTTConfig:
    """Configuration for Test-Time Training."""
    
    # Learning rate for inner loop adaptation
    # TODO: Consider adding LR warmup/decay for multi-epoch TTT
    inner_lr: float = 1e-3
    
    # Number of epochs over augmented samples (full passes through data)
    inner_epochs: int = 2
    
    # Number of augmented examples to generate from support set
    num_augmentations: int = 50
    
    # Batch size for inner loop (samples per step)
    inner_batch_size: int = 8
    
    # Gradient clipping (max norm)
    max_grad_norm: float = 1.0
    
    # Whether to use loss only on output grid pixels
    loss_on_output_only: bool = True
    
    # Max sequence length for tokenization
    max_seq_length: int = 8192


@dataclass
class InferenceConfig:
    """Configuration for inference with thinking and verification."""
    
    # Number of candidate outputs for best-of-N sampling
    num_candidates: int = 3
    
    # Sampling temperature (0 = greedy, higher = more random)
    temperature: float = 0.7
    
    # Top-p nucleus sampling threshold
    top_p: float = 0.9
    
    # Whether to add <think> prompt for reasoning
    use_thinking: bool = True
    
    # Maximum new tokens to generate
    max_new_tokens: int = 512
    
    # Stop generation at these tokens
    stop_tokens: Tuple[str, ...] = ("<|answer|>", "<|endoftext|>")
    
    # Thinking prompt to inject
    thinking_prompt: str = "<think>Let me analyze this step by step.\n"


@dataclass
class LoRATTTConfig:
    """Combined config for full LoRA + TTT pipeline."""
    
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    ttt: TTTConfig = field(default_factory=TTTConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Model settings
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    dtype: str = "bfloat16"
    
    # Checkpoint paths
    base_checkpoint: str = ""  # Phase 1-3 finetuned model
    lora_checkpoint: str = ""  # Trained LoRA weights (optional)
    
    @property
    def torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.dtype]


@dataclass
class LoRATrainConfig:
    """Configuration for LoRA training.
    
    Edit defaults directly in this class before running training.
    """
    
    # === Data ===
    data_dir: str = ""  # Directory containing training data shards
    eval_dir: Optional[str] = None  # Optional: evaluation data directory
    
    # === Model ===
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    base_checkpoint: str = ""  # Path to phase 1-3 finetuned checkpoint
    
    # === LoRA ===
    lora_r: int = 4  # LoRA rank (4 is a good default)
    lora_alpha: float = 1.0  # Scaling = alpha / r
    lora_dropout: float = 0.0  # Dropout on LoRA path
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj")  # Layers to adapt
    
    # === Training ===
    batch_size: int = 4  # Per-GPU batch size
    grad_accum_steps: int = 8  # Effective batch = batch_size * grad_accum_steps = 32
    epochs: int = 3
    max_steps: Optional[int] = None  # Stop after N steps (overrides epochs if set)
    lr: float = 1e-4  # Learning rate
    warmup_steps: int = 100  # Linear warmup steps
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # Gradient clipping
    max_seq_length: int = 8192  # Max sequence length
    
    # === Loss ===
    loss_on_output_only: bool = True  # Only compute loss on output grid tokens
    fim_ratio: float = 0.0  # Fraction of samples to use Fill-In-Middle objective
    
    # === Checkpointing ===
    output_dir: str = "./checkpoints/lora/"
    save_steps: int = 200  # Save checkpoint every N steps
    log_steps: int = 10  # Log metrics every N steps
    eval_steps: int = 200  # Evaluate every N steps
    eval_samples: int = 200  # Number of samples for evaluation
    
    # === Hardware ===
    dtype: str = "bfloat16"  # float32, float16, or bfloat16
    use_amp: bool = True  # Automatic mixed precision
    num_workers: int = 0  # DataLoader workers (0 = main process)
    use_torch_compile: bool = True  # Use torch.compile for faster training
    compile_mode: str = "default"  # torch.compile mode: default, reduce-overhead, max-autotune
    gradient_checkpointing: bool = True  # Trade compute for memory (required for large batches)
    
    # === W&B ===
    use_wandb: bool = True
    wandb_project: str = "arc_lora"
    wandb_run_name: Optional[str] = None
    
    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.dtype]



@dataclass
class EvalConfig:
    """Configuration for evaluation. Edit values here before running."""
    
    # === Paths ===
    dataset_path: str = ""  # Path to ARC dataset (directory or file)
    base_checkpoint: str = ""  # Path to base model checkpoint (phase 2/3)
    lora_checkpoint: str = ""  # Path to trained LoRA weights (optional, can be empty)
    
    # === Model ===
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    dtype: str = "bfloat16"
    
    # === LoRA ===
    lora_r: int = 4
    lora_alpha: float = 1.0
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj")
    
    # === TTT ===
    ttt_lr: float = 1e-3  # Inner loop learning rate
    ttt_epochs: int = 1  # Number of epochs over augmented samples
    num_augmentations: int = 50  # Augmented examples per puzzle
    ttt_batch_size: int = 8  # Batch size for TTT
    
    # === Inference ===
    num_candidates: int = 3  # Best-of-N sampling
    temperature: float = 0.7  # Sampling temperature (0 = greedy)
    top_p: float = 0.9  # Top-p nucleus sampling
    use_thinking: bool = True  # Add <think> prompt
    max_new_tokens: int = 512  # Max tokens to generate
    
    # === Evaluation ===
    max_puzzles: Optional[int] = None  # Limit number of puzzles (None = all)
    
    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.dtype]