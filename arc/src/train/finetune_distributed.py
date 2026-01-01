"""
ARC-AGI Distributed Finetuning Script

Supports both single-GPU and multi-GPU (FSDP) training.
Includes Weights & Biases logging for experiment tracking.

Usage:
    # Single GPU
    from src.train.finetune_distributed import train
    from src.train.config import Phase1Config
    
    config = Phase1Config(data_dir="/path/to/data")
    train(config)
    
    # Multi-GPU with torchrun
    # In your script, set config.distributed = True, then run with:
    # torchrun --nproc_per_node=8 -m src.train.finetune_distributed
"""

import os
import time

# Silence tokenizer parallelism warning (must be before imports)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Any
from dataclasses import asdict

# Load .env file (for WANDB_API_KEY, etc.)
from dotenv import load_dotenv
load_dotenv()

from src.data.tokenizer_2d import Arc2DTokenizer
from src.data.dataloader import ArcDataset, collate_arc_2d
from src.model.qwen_2d import load_qwen_2d
from src.train.config import TrainConfig, Phase1Config, Phase2Config, Phase3Config
from src.train.loss import compute_ntp_loss
from src.train.distributed import (
    setup_distributed,
    cleanup_distributed,
    wrap_model_fsdp,
    save_fsdp_checkpoint,
    load_fsdp_checkpoint,
    DistributedConfig,
    is_main_process,
    get_world_size,
    get_rank,
    barrier,
    print_rank0,
    all_reduce_mean,
)


# === Weights & Biases Setup ===

def init_wandb(config: TrainConfig, world_size: int) -> Optional[Any]:
    """
    Initialize Weights & Biases logging (only on rank 0).
    
    Returns:
        wandb run object or None if disabled/not main process
    """
    if not config.use_wandb or not is_main_process():
        return None
    
    try:
        import wandb
        
        # Generate run name if not provided
        run_name = config.wandb_run_name
        if run_name is None:
            run_name = f"finetune-phase{config.phase}_lr{config.lr}_bs{config.batch_size * config.grad_accum_steps * world_size}"
        
        # Convert config to dict for logging
        config_dict = asdict(config)
        config_dict["effective_batch_size"] = config.batch_size * config.grad_accum_steps * world_size
        config_dict["world_size"] = world_size
        
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            group=f"finetune-phase{config.phase}",  # Group runs by phase
            tags=[f"finetune-phase{config.phase}", "2d-rope", "full-finetune"],
            config=config_dict,
            resume="allow",
        )
        
        print_rank0(f"W&B initialized: {run.url}")
        return run
    
    except ImportError:
        print_rank0("Warning: wandb not installed, skipping W&B logging")
        return None
    except Exception as e:
        print_rank0(f"Warning: Failed to initialize W&B: {e}")
        return None


def log_wandb(
    wandb_run: Optional[Any],
    metrics: dict,
    step: int,
) -> None:
    """Log metrics to W&B if enabled."""
    if wandb_run is not None:
        import wandb
        wandb.log(metrics, step=step)


def finish_wandb(wandb_run: Optional[Any]) -> None:
    """Finish W&B run if enabled."""
    if wandb_run is not None:
        import wandb
        wandb.finish()


def get_optimizer(model, config: TrainConfig, use_fsdp: bool = False):
    """
    Get optimizer - 8-bit AdamW if configured, else standard AdamW.
    
    Note: 8-bit Adam may not work well with FSDP due to state sharding.
    """
    # 8-bit Adam doesn't work well with FSDP
    if config.use_8bit_adam and not use_fsdp:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.95),
            )
            print_rank0("Using 8-bit AdamW (bitsandbytes)")
            return optimizer
        except ImportError:
            print_rank0("Warning: bitsandbytes not installed, falling back to standard AdamW")
    
    if use_fsdp and config.use_8bit_adam:
        print_rank0("Note: 8-bit AdamW disabled for FSDP compatibility, using standard AdamW")
    
    return AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )


def create_distributed_dataloader(
    data_dir: str,
    tokenizer: Any,
    config: TrainConfig,
    is_train: bool = True,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """
    Create a DataLoader that works for both single and multi-GPU training.
    
    For multi-GPU: Uses DistributedSampler to shard data across GPUs.
    """
    dataset = ArcDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        shuffle_shards=is_train and world_size == 1,  # Don't shuffle if using DistributedSampler
        fim_ratio=config.fim_ratio if is_train else 0.0,
        max_seq_length=config.max_seq_length,
    )
    
    # For distributed training, use DistributedSampler
    # Note: IterableDataset doesn't use samplers the same way
    # The dataset already handles shard distribution via worker_info
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_arc_2d,
        pin_memory=True,
    )


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    config: TrainConfig,
    global_step: int,
    epoch: int,
    use_fsdp: bool = False,
    wandb_run: Optional[Any] = None,
    save_checkpoint_fn: Optional[callable] = None,
    eval_dataloader: Optional[DataLoader] = None,
    best_eval_loss: float = float("inf"),
) -> tuple[int, float, float]:
    """
    Train for one epoch.
    
    Handles both single-GPU and FSDP training.
    Runs evaluation every config.eval_steps if eval_dataloader is provided.
    
    Args:
        model: The model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: LR scheduler
        device: Device to train on
        config: Training config
        global_step: Current global step
        epoch: Current epoch number
        use_fsdp: Whether using FSDP
        wandb_run: W&B run object for logging (or None)
        save_checkpoint_fn: Function to call for saving checkpoints
        eval_dataloader: Optional eval dataloader for mid-epoch evaluation
        best_eval_loss: Current best eval loss (for tracking improvement)
    
    Returns:
        (final_step, avg_loss, best_eval_loss)
    """
    model.train()
    total_loss = 0.0
    step_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    # Only show progress bar on rank 0
    pbar = tqdm(dataloader, desc="Training", disable=not is_main_process())
    
    rank = get_rank()
    step_start_time = None
    
    for batch_idx, batch in enumerate(pbar):
        # Start timing when we begin a new step (first batch in accumulation)
        if batch_idx % config.grad_accum_steps == 0:
            step_start_time = time.time()
        
        # seq_len = batch["input_ids"].shape[1]
        # batch_size = batch["input_ids"].shape[0]
        # print(f'[Rank {rank}] Batch {batch_idx}: seq_len={seq_len}, batch_size={batch_size}', flush=True)
        
        # print(f'[Rank {rank}] Calling compute_ntp_loss...', flush=True)
        loss = compute_ntp_loss(
            model, batch, device,
            use_amp=config.use_amp,
            amp_dtype=config.torch_dtype,
            loss_on_output_only=config.loss_on_output_only,
        )
        # print(f'[Rank {rank}] compute_ntp_loss returned, loss={loss.item():.4f}', flush=True)
        
        # Clean up memory before backward
        if device.type == "cuda":
            torch.cuda.synchronize()
            # print(f'[Rank {rank}] CUDA synchronized before backward', flush=True)
        
        loss = loss / config.grad_accum_steps
        
        # Sync all ranks before backward to avoid FSDP deadlock
        if use_fsdp:
            # print(f'[Rank {rank}] Barrier before backward...', flush=True)
            barrier()
            # print(f'[Rank {rank}] Barrier passed, starting backward...', flush=True)
        
        # print(f'[Rank {rank}] Starting backward()...', flush=True)
        # print(f'[Rank {rank}] Loss tensor: requires_grad={loss.requires_grad}, device={loss.device}, shape={loss.shape}', flush=True)
        
        try:
            loss.backward()
            # print(f'[Rank {rank}] backward() completed successfully', flush=True)
        except Exception as e:
            # print(f'[Rank {rank}] backward() FAILED: {e}', flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        step_loss += loss.item()
        del loss
        
        if (batch_idx + 1) % config.grad_accum_steps == 0:
            # print(f'[Rank {rank}] Accumulation complete, computing grad norm...', flush=True)
            # Compute gradient norm before clipping (for logging)
            if use_fsdp:
                # FSDP: clip_grad_norm_ returns the total norm
                grad_norm = model.clip_grad_norm_(config.max_grad_norm)
            else:
                # Compute norm, then clip
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            # print(f'[Rank {rank}] Grad norm computed: {grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm:.4f}', flush=True)
            
            # print(f'[Rank {rank}] Calling optimizer.step()...', flush=True)
            optimizer.step()
            # print(f'[Rank {rank}] optimizer.step() completed', flush=True)
            
            # print(f'[Rank {rank}] Calling scheduler.step()...', flush=True)
            scheduler.step()
            # print(f'[Rank {rank}] scheduler.step() completed', flush=True)
            
            # print(f'[Rank {rank}] Calling optimizer.zero_grad()...', flush=True)
            optimizer.zero_grad()
            # print(f'[Rank {rank}] optimizer.zero_grad() completed', flush=True)
            
            # Clear cache periodically
            if device.type == "cuda" and (batch_idx + 1) % (config.grad_accum_steps * 10) == 0:
                torch.cuda.empty_cache()
            
            global_step += 1
            
            # Aggregate loss across all GPUs for logging
            # print(f'[Rank {rank}] Aggregating loss across GPUs...', flush=True)
            batch_loss = step_loss * config.grad_accum_steps
            if use_fsdp:
                # print(f'[Rank {rank}] Calling all_reduce_mean for loss...', flush=True)
                batch_loss_tensor = torch.tensor(batch_loss, device=device)
                batch_loss = all_reduce_mean(batch_loss_tensor).item()
                # print(f'[Rank {rank}] all_reduce_mean completed, batch_loss={batch_loss:.4f}', flush=True)
            
            total_loss += batch_loss
            # print(f'[Rank {rank}] Loss aggregation done, moving to logging...', flush=True)
            num_batches += 1
            lr = scheduler.get_last_lr()[0]
            
            # Convert grad_norm to float for logging
            grad_norm_float = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
            
            # Calculate step time (time for all batches in this accumulation step)
            step_time = time.time() - step_start_time if step_start_time else 0
            # Log to console and W&B
            if global_step % config.log_steps == 0:
                if is_main_process():
                    print(f'[Step {global_step}] loss={batch_loss:.4f}, step_time={step_time:.2f}s', flush=True)
                    pbar.set_postfix({
                        "step": global_step,
                        "loss": f"{batch_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "grad": f"{grad_norm_float:.2f}",
                        "step_time": f"{step_time:.2f}s"
                    })
                
                # Log to W&B
                log_wandb(wandb_run, {
                    "train/loss": batch_loss,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm_float,
                    "train/step_time": step_time,
                    "train/epoch": epoch,
                }, step=global_step)
            
            # Save checkpoint every save_steps
            if global_step % config.save_steps == 0 and save_checkpoint_fn is not None:
                save_checkpoint_fn(epoch, global_step, batch_loss, f"step_{global_step}")
                log_wandb(wandb_run, {"checkpoint/step": global_step}, step=global_step)
            
            # Mid-epoch evaluation every eval_steps
            if eval_dataloader is not None and global_step % config.eval_steps == 0:
                eval_loss = evaluate(model, eval_dataloader, device, config, use_fsdp)
                print_rank0(f"\n[Step {global_step}] Eval loss: {eval_loss:.4f}")
                
                log_wandb(wandb_run, {
                    "eval/loss": eval_loss,
                    "eval/step": global_step,
                }, step=global_step)
                
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    if save_checkpoint_fn is not None:
                        save_checkpoint_fn(epoch, global_step, eval_loss, "best")
                    log_wandb(wandb_run, {"eval/best_loss": best_eval_loss}, step=global_step)
                
                model.train()  # Switch back to training mode
            
            step_loss = 0.0
            # print(f'[Rank {rank}] Step {global_step} complete, moving to next batch...', flush=True)
            
            if config.max_steps and global_step >= config.max_steps:
                print(f'[Rank {rank}] Reached max_steps, breaking...', flush=True)
                break
    
    print(f'[Rank {rank}] Epoch complete, computing avg_loss...', flush=True)
    avg_loss = total_loss / max(num_batches, 1)
    print(f'[Rank {rank}] train_epoch returning: step={global_step}, avg_loss={avg_loss:.4f}', flush=True)
    return global_step, avg_loss, best_eval_loss


def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    config: TrainConfig,
    use_fsdp: bool = False,
    max_samples: Optional[int] = None,
) -> float:
    """
    Evaluate model on validation set.
    
    Args:
        max_samples: If set, only evaluate on this many samples (for quick eval).
                    None = evaluate on entire dataset.
    
    Returns average loss (aggregated across GPUs if distributed).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    num_samples = 0
    
    # Use max_samples from config if not explicitly provided
    if max_samples is None:
        max_samples = config.eval_samples if hasattr(config, 'eval_samples') and config.eval_samples else None
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not is_main_process()):
            loss = compute_ntp_loss(
                model, batch, device,
                use_amp=config.use_amp,
                amp_dtype=config.torch_dtype,
                loss_on_output_only=config.loss_on_output_only,
            )
            total_loss += loss.item()
            num_batches += 1
            num_samples += batch["input_ids"].size(0)
            
            # Stop early if we've evaluated enough samples
            if max_samples and num_samples >= max_samples:
                break
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # Aggregate across GPUs
    if use_fsdp:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        avg_loss = all_reduce_mean(avg_loss_tensor).item()
    
    return avg_loss


def train(config: TrainConfig) -> None:
    """
    Main training function.
    
    Automatically handles single-GPU and multi-GPU (FSDP) training.
    """
    # === Setup Distributed (if enabled) ===
    use_fsdp = config.distributed
    rank, world_size, local_rank = 0, 1, 0
    
    if use_fsdp:
        rank, world_size, local_rank = setup_distributed()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[Rank {rank}] Distributed setup complete, device: {device}", flush=True)
    
    # === Validate and Print Config ===
    config.validate()
    
    print_rank0("=" * 60)
    print_rank0(f"ARC-AGI Finetuning - Phase {config.phase}")
    print_rank0(f"Device: {device}, Dtype: {config.torch_dtype}")
    print_rank0(f"Distributed: {use_fsdp}, World Size: {world_size}")
    print_rank0("=" * 60)
    
    # === Create Output Directory ===
    output_dir = Path(config.output_dir)
    # All ranks try to create (exist_ok=True is safe) - avoids barrier which can hang
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === Initialize Tokenizer ===
    print(f"[Rank {rank}] Initializing tokenizer...", flush=True)
    tokenizer_2d = Arc2DTokenizer(model_name=config.model_name)
    print(f"[Rank {rank}] Tokenizer ready", flush=True)
    
    # === Load Model ===
    # === Load Model ===
    # All ranks load model to CPU, FSDP will shard to GPUs
    # Using low_cpu_mem_usage in load_qwen_2d helps reduce peak memory
    print(f"[Rank {rank}] Loading model to CPU...", flush=True)
    model, tokenizer_2d = load_qwen_2d(
        model_name=config.model_name,
        tokenizer_2d=tokenizer_2d,
        checkpoint_path=None,
        dtype=config.torch_dtype,
        device="cpu",  # Load to CPU, FSDP will shard to GPUs
    )
    print(f"[Rank {rank}] Model loaded", flush=True)
    
    # === Apply torch.compile (if enabled, must be before FSDP) ===
    if config.use_torch_compile:
        print_rank0(f"Compiling model with mode='{config.compile_mode}'...")
        print_rank0("Note: Variable-length sequences require dynamic shapes")
        print_rank0("First forward pass will be slower (compilation), then faster")
        try:
            # Use dynamic=True for variable-length sequences (padded batches)
            # This allows torch.compile to handle different sequence lengths
            model = torch.compile(
                model, 
                mode=config.compile_mode,
                dynamic=True,  # Required for variable-length sequences
            )
            print_rank0("Model compiled successfully with dynamic shapes")
        except Exception as e:
            print_rank0(f"Warning: torch.compile failed: {e}")
            print_rank0("Continuing without compilation. Consider disabling use_torch_compile.")
            # Don't re-raise - continue without compilation
    
    # === Apply FSDP Wrapping (if distributed) ===
    if use_fsdp:
        print(f"[Rank {rank}] Starting FSDP wrap...", flush=True)
        
        # Get the transformer layer class for wrapping policy
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        
        fsdp_config = DistributedConfig(
            sharding_strategy=config.sharding_strategy,
            mixed_precision="bf16" if config.dtype == "bfloat16" else config.dtype,
            cpu_offload=config.cpu_offload,
            activation_checkpointing=config.gradient_checkpointing,
            sync_module_states=False,  # All ranks have same weights, no need to sync
        )
        
        model = wrap_model_fsdp(
            model,
            fsdp_config,
            transformer_layer_cls=Qwen2DecoderLayer,
            device_id=local_rank,
        )
        print_rank0("Model wrapped with FSDP")
    else:
        # Single GPU: move model to device and enable gradient checkpointing
        model = model.to(device)
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            print_rank0("Gradient checkpointing enabled")
    
    # === Create DataLoaders ===
    train_dataloader = create_distributed_dataloader(
        data_dir=config.data_dir,
        tokenizer=tokenizer_2d,
        config=config,
        is_train=True,
        world_size=world_size,
        rank=rank,
    )
    
    eval_dataloader = None
    if config.eval_dir:
        eval_dataloader = create_distributed_dataloader(
            data_dir=config.eval_dir,
            tokenizer=tokenizer_2d,
            config=config,
            is_train=False,
            world_size=world_size,
            rank=rank,
        )
    
    # === Setup Optimizer ===
    optimizer = get_optimizer(model, config, use_fsdp=use_fsdp)
    
    # === Setup Scheduler ===
    total_steps = config.max_steps if config.max_steps else config.epochs * 10000
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.warmup_steps,
    )
    
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - config.warmup_steps,
        eta_min=config.lr * 0.1,
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[config.warmup_steps],
    )
    
    # === Load Checkpoint (if resuming) ===
    global_step = 0
    start_epoch = 0
    
    if config.checkpoint:
        if use_fsdp:
            metadata = load_fsdp_checkpoint(
                config.checkpoint, model, optimizer, scheduler
            )
        else:
            from src.model.qwen_2d import load_checkpoint
            metadata = load_checkpoint(config.checkpoint, model, optimizer, scheduler)
        
        global_step = metadata["step"]
        start_epoch = metadata["epoch"]
        print_rank0(f"Resumed from checkpoint: epoch {start_epoch}, step {global_step}")
    
    # === Initialize W&B ===
    effective_batch_size = config.batch_size * config.grad_accum_steps * world_size
    wandb_run = init_wandb(config, world_size)
    
    # === Print Training Info ===
    print_rank0(f"\nStarting training...")
    print_rank0(f"  Epochs: {config.epochs}")
    print_rank0(f"  Batch size per GPU: {config.batch_size}")
    print_rank0(f"  Gradient accumulation: {config.grad_accum_steps}")
    print_rank0(f"  World size: {world_size}")
    print_rank0(f"  Effective batch size: {effective_batch_size}")
    print_rank0(f"  Learning rate: {config.lr}")
    print_rank0(f"  FIM ratio: {config.fim_ratio:.0%}")
    print_rank0(f"  Loss on output only: {config.loss_on_output_only}")
    print_rank0(f"  Save every: {config.save_steps} steps")
    print_rank0(f"  Eval every: {config.eval_steps} steps ({config.eval_samples} samples)")
    print_rank0(f"  W&B logging: {wandb_run is not None}")
    print_rank0()
    
    # === Checkpoint Saving Helper ===
    def save_checkpoint_helper(epoch: int, step: int, loss: float, name: str, save_optimizer: bool = None):
        """
        Helper to save checkpoints (handles FSDP vs single GPU).
        
        Args:
            save_optimizer: Override config.save_optimizer_state for this checkpoint.
                           None = use config default.
        """
        ckpt_path = output_dir / f"phase{config.phase}_{name}.pt"
        should_save_optimizer = save_optimizer if save_optimizer is not None else config.save_optimizer_state
        
        if use_fsdp:
            save_fsdp_checkpoint(
                model, optimizer, scheduler,
                epoch, step, loss,
                str(ckpt_path), rank,
                save_optimizer_state=should_save_optimizer,
            )
        else:
            from src.model.qwen_2d import save_checkpoint
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, step, loss,
                str(ckpt_path), config.phase,
                save_optimizer_state=should_save_optimizer,
            )
    
    # === Training Loop ===
    best_eval_loss = float("inf")
    train_loss = float("inf")
    
    try:
        for epoch in range(start_epoch, config.epochs):
            print_rank0(f"\n{'=' * 40}")
            print_rank0(f"Epoch {epoch + 1}/{config.epochs}")
            print_rank0(f"{'=' * 40}")
            
            # Pass eval_dataloader for mid-epoch evaluation every eval_steps
            global_step, train_loss, best_eval_loss = train_epoch(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                config=config,
                global_step=global_step,
                epoch=epoch + 1,
                use_fsdp=use_fsdp,
                wandb_run=wandb_run,
                save_checkpoint_fn=save_checkpoint_helper,
                eval_dataloader=eval_dataloader,
                best_eval_loss=best_eval_loss,
            )
            
            print_rank0(f"\nEpoch {epoch + 1} complete. Train loss: {train_loss:.4f}")
            if best_eval_loss < float("inf"):
                print_rank0(f"Best eval loss so far: {best_eval_loss:.4f}")
            
            # Log epoch metrics to W&B
            log_wandb(wandb_run, {
                "epoch/train_loss": train_loss,
                "epoch/number": epoch + 1,
            }, step=global_step)
            
            # === Save Epoch Checkpoint (skip if single epoch - redundant with step checkpoints) ===
            if config.epochs > 1:
                save_checkpoint_helper(epoch + 1, global_step, train_loss, f"epoch{epoch + 1}")
            
            if config.max_steps and global_step >= config.max_steps:
                print_rank0(f"Reached max_steps ({config.max_steps}), stopping.")
                break
        
        # === Final Save (skip if single epoch - last step checkpoint is sufficient) ===
        if config.epochs > 1:
            save_checkpoint_helper(config.epochs, global_step, train_loss, "final")
            print_rank0(f"\n{'=' * 60}")
            print_rank0("Training complete!")
            print_rank0(f"Final checkpoint: {output_dir / f'phase{config.phase}_final.pt'}")
        else:
            print_rank0(f"\n{'=' * 60}")
            print_rank0("Training complete!")
            print_rank0(f"Latest checkpoint: phase{config.phase}_step_*.pt")
        print_rank0(f"{'=' * 60}")
    
    finally:
        # === Cleanup ===
        finish_wandb(wandb_run)
        if use_fsdp:
            cleanup_distributed()


def main():
    """
    Example main function - customize config as needed.
    
    For multi-GPU, run with:
        NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python -m torch.distributed.run --nproc_per_node=8 -m src.train.finetune_distributed
    """
    # === Configure your training here ===
    config = Phase1Config(
        data_dir="/root/arc_data/train_data",
        eval_dir="/root/arc_data/eval_data",
        distributed=True,  # Set to True for multi-GPU
        batch_size=1,  # batch=2 OOMs on 11K sequences during backward
        grad_accum_steps=4,  # Effective batch = 1 × 4 × 8 = 32
        lr=2e-5,
        max_steps=20_000,
        max_seq_length=1024 * 10,  # Chunked CE handles sequences up to this
    )
    
    train(config)


if __name__ == "__main__":
    main()

