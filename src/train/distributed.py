"""
Distributed training utilities for FSDP.

Provides reusable components for multi-GPU training:
- Process group setup/teardown
- FSDP wrapping with optimal policies
- Distributed checkpointing
- Memory-efficient gradient handling

Usage:
    from src.train.distributed import (
        setup_distributed,
        cleanup_distributed,
        wrap_model_fsdp,
        save_fsdp_checkpoint,
        load_fsdp_checkpoint,
    )
"""

import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    StateDictType,
)
from functools import partial
from typing import Optional, Dict, Any, Type
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    
    # Basic distributed settings
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    
    # FSDP sharding strategy
    # FULL_SHARD: Shard parameters, gradients, and optimizer states
    # SHARD_GRAD_OP: Shard gradients and optimizer states only
    # NO_SHARD: DDP-like behavior (no sharding)
    sharding_strategy: str = "FULL_SHARD"
    
    # Mixed precision policy
    # "bf16": bfloat16 for compute, fp32 for reductions
    # "fp16": float16 for compute, fp32 for reductions
    # "fp32": no mixed precision
    mixed_precision: str = "bf16"
    
    # CPU offload (trades speed for memory)
    cpu_offload: bool = False
    
    # Backward prefetch (overlap communication with compute)
    backward_prefetch: str = "BACKWARD_PRE"  # or "BACKWARD_POST" or None
    
    # Activation checkpointing (recompute instead of store)
    activation_checkpointing: bool = True
    
    # Sync module states on wrap (ensures all ranks start with same weights)
    sync_module_states: bool = True
    
    # Use original parameters for optimizer (required for some optimizers)
    use_orig_params: bool = True


def setup_distributed(
    backend: str = "nccl",
    timeout_minutes: int = 30,
) -> tuple[int, int, int]:
    """
    Initialize distributed training environment.
    
    Expects environment variables set by torchrun/torch.distributed.launch:
    - RANK: Global rank of this process
    - WORLD_SIZE: Total number of processes
    - LOCAL_RANK: Rank within this node
    - MASTER_ADDR: Address of rank 0
    - MASTER_PORT: Port for communication
    
    Returns:
        (rank, world_size, local_rank)
    """
    # Check if already initialized
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    
    # Get distributed info from environment
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if world_size == 1:
        print("Single GPU detected, skipping distributed setup")
        return rank, world_size, local_rank
    
    # Set CUDA device before init
    torch.cuda.set_device(local_rank)
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=torch.distributed.default_pg_timeout if timeout_minutes is None 
                else torch.timedelta(minutes=timeout_minutes),
    )
    
    # Synchronize all processes
    dist.barrier()
    
    if rank == 0:
        print(f"Distributed training initialized:")
        print(f"  World size: {world_size}")
        print(f"  Backend: {backend}")
    
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get global rank of current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def _get_sharding_strategy(strategy: str) -> ShardingStrategy:
    """Convert string to ShardingStrategy enum."""
    strategies = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown sharding strategy: {strategy}. Options: {list(strategies.keys())}")
    return strategies[strategy]


def _get_mixed_precision_policy(precision: str) -> Optional[MixedPrecision]:
    """Create MixedPrecision policy from string."""
    if precision == "fp32":
        return None
    
    if precision == "bf16":
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    
    if precision == "fp16":
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    
    raise ValueError(f"Unknown precision: {precision}. Options: fp32, fp16, bf16")


def _get_backward_prefetch(prefetch: Optional[str]) -> Optional[BackwardPrefetch]:
    """Convert string to BackwardPrefetch enum."""
    if prefetch is None:
        return None
    
    prefetch_options = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
    }
    
    if prefetch not in prefetch_options:
        raise ValueError(f"Unknown backward prefetch: {prefetch}. Options: {list(prefetch_options.keys())}")
    
    return prefetch_options[prefetch]


def get_transformer_wrap_policy(transformer_layer_cls: Type) -> Any:
    """
    Create an auto wrap policy for transformer models.
    
    Wraps at the transformer layer level (e.g., Qwen2DecoderLayer).
    This ensures each layer is a separate FSDP unit.
    
    Args:
        transformer_layer_cls: The decoder layer class to wrap at
                              (e.g., Qwen2DecoderLayer from transformers)
    
    Returns:
        Auto wrap policy function
    """
    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_cls},
    )


def wrap_model_fsdp(
    model: torch.nn.Module,
    config: DistributedConfig,
    transformer_layer_cls: Optional[Type] = None,
    device_id: Optional[int] = None,
) -> FSDP:
    """
    Wrap a model with FSDP for distributed training.
    
    Args:
        model: The model to wrap (should NOT be on GPU yet for best memory efficiency)
        config: DistributedConfig with FSDP settings
        transformer_layer_cls: The transformer layer class for auto-wrapping
                               If None, uses size-based policy
        device_id: CUDA device ID (usually local_rank)
    
    Returns:
        FSDP-wrapped model
    """
    # Get device
    if device_id is None:
        device_id = int(os.environ.get("LOCAL_RANK", 0))
    
    # Build FSDP config
    sharding_strategy = _get_sharding_strategy(config.sharding_strategy)
    mixed_precision = _get_mixed_precision_policy(config.mixed_precision)
    backward_prefetch = _get_backward_prefetch(config.backward_prefetch)
    cpu_offload = CPUOffload(offload_params=True) if config.cpu_offload else None
    
    # Build auto wrap policy
    if transformer_layer_cls is not None:
        auto_wrap_policy = get_transformer_wrap_policy(transformer_layer_cls)
    else:
        # Fallback: wrap modules with >100M parameters
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=100_000_000,
        )
    
    # Wrap model
    fsdp_model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        backward_prefetch=backward_prefetch,
        cpu_offload=cpu_offload,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.device("cuda", device_id),
        sync_module_states=config.sync_module_states,
        use_orig_params=config.use_orig_params,
    )
    
    # Enable activation checkpointing if requested
    if config.activation_checkpointing:
        _apply_activation_checkpointing(fsdp_model, transformer_layer_cls)
    
    return fsdp_model


def _apply_activation_checkpointing(
    model: FSDP,
    transformer_layer_cls: Optional[Type] = None,
) -> None:
    """
    Apply activation checkpointing to transformer layers.
    
    This recomputes activations during backward pass instead of storing them,
    trading compute for memory.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )
    
    if transformer_layer_cls is None:
        # Can't apply without knowing the layer type
        return
    
    # Define check function for which modules to checkpoint
    def check_fn(module: torch.nn.Module) -> bool:
        return isinstance(module, transformer_layer_cls)
    
    # Apply checkpointing
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=check_fn,
    )


def save_fsdp_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    loss: float,
    path: str,
    rank: int = 0,
) -> None:
    """
    Save FSDP checkpoint.
    
    Gathers full state dict to rank 0 and saves.
    Only rank 0 actually writes to disk.
    
    Args:
        model: FSDP-wrapped model
        optimizer: Optimizer
        scheduler: LR scheduler
        epoch: Current epoch
        step: Current step
        loss: Current loss
        path: Save path
        rank: Current process rank
    """
    # Configure state dict gathering
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model_state = model.state_dict()
        
        # Only gather optimizer state on rank 0
        if rank == 0:
            optim_state = FSDP.optim_state_dict(model, optimizer)
        else:
            optim_state = None
    
    # Only rank 0 saves
    if rank == 0:
        checkpoint = {
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "step": step,
            "loss": loss,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    # Synchronize all ranks
    barrier()


def load_fsdp_checkpoint(
    path: str,
    model: FSDP,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load FSDP checkpoint.
    
    All ranks load the full state dict, then FSDP handles sharding.
    
    Args:
        path: Checkpoint path
        model: FSDP-wrapped model
        optimizer: Optimizer (optional)
        scheduler: LR scheduler (optional)
        strict: Whether to strictly enforce state dict key matching
    
    Returns:
        Metadata dict with epoch, step, loss
    """
    # All ranks load (FSDP will handle distribution)
    checkpoint = torch.load(path, map_location="cpu")
    
    # Load model state
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    # Load optimizer state if provided
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            optim_state = FSDP.optim_state_dict_to_load(
                model, optimizer, checkpoint["optimizer_state_dict"]
            )
            optimizer.load_state_dict(optim_state)
    
    # Load scheduler state
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    if is_main_process():
        print(f"Loaded checkpoint from {path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Step: {checkpoint.get('step', 'N/A')}")
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", float("inf")),
    }


def print_rank0(*args, **kwargs) -> None:
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce a tensor and compute mean across all ranks.
    
    Useful for aggregating metrics like loss across GPUs.
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor
    
    # Clone to avoid modifying original
    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= get_world_size()
    return reduced

