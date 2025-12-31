"""
ARC-AGI Distributed Finetuning Script

Supports both single-GPU and multi-GPU (FSDP) training.

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
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Any

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
    use_fsdp: bool = False,
) -> tuple[int, float]:
    """
    Train for one epoch.
    
    Handles both single-GPU and FSDP training.
    
    Returns:
        (final_step, avg_loss)
    """
    model.train()
    total_loss = 0.0
    step_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    # Only show progress bar on rank 0
    pbar = tqdm(dataloader, desc="Training", disable=not is_main_process())
    
    for batch_idx, batch in enumerate(pbar):
        loss = compute_ntp_loss(
            model, batch, device,
            use_amp=config.use_amp,
            amp_dtype=config.torch_dtype,
            loss_on_output_only=config.loss_on_output_only,
        )
        loss = loss / config.grad_accum_steps
        loss.backward()
        
        step_loss += loss.item()
        del loss
        
        if (batch_idx + 1) % config.grad_accum_steps == 0:
            # Gradient clipping
            if use_fsdp:
                # FSDP handles gradient clipping differently
                model.clip_grad_norm_(config.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Clear cache periodically
            if device.type == "cuda" and (batch_idx + 1) % (config.grad_accum_steps * 10) == 0:
                torch.cuda.empty_cache()
            
            global_step += 1
            
            # Aggregate loss across all GPUs for logging
            batch_loss = step_loss * config.grad_accum_steps
            if use_fsdp:
                batch_loss_tensor = torch.tensor(batch_loss, device=device)
                batch_loss = all_reduce_mean(batch_loss_tensor).item()
            
            total_loss += batch_loss
            num_batches += 1
            
            if global_step % config.log_steps == 0 and is_main_process():
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "step": global_step,
                    "loss": f"{batch_loss:.4f}",
                    "lr": f"{lr:.2e}"
                })
            
            step_loss = 0.0
            
            if config.max_steps and global_step >= config.max_steps:
                break
    
    avg_loss = total_loss / max(num_batches, 1)
    return global_step, avg_loss


def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    config: TrainConfig,
    use_fsdp: bool = False,
) -> float:
    """
    Evaluate model on validation set.
    
    Returns average loss (aggregated across GPUs if distributed).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
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
    
    # === Validate and Print Config ===
    config.validate()
    
    print_rank0("=" * 60)
    print_rank0(f"ARC-AGI Finetuning - Phase {config.phase}")
    print_rank0(f"Device: {device}, Dtype: {config.torch_dtype}")
    print_rank0(f"Distributed: {use_fsdp}, World Size: {world_size}")
    print_rank0("=" * 60)
    
    # === Create Output Directory ===
    output_dir = Path(config.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier()  # Wait for directory creation
    
    # === Initialize Tokenizer ===
    tokenizer_2d = Arc2DTokenizer(model_name=config.model_name)
    
    # === Load Model ===
    # For FSDP, load to CPU first, then wrap
    load_device = "cpu" if use_fsdp else str(device)
    
    model, tokenizer_2d = load_qwen_2d(
        model_name=config.model_name,
        tokenizer_2d=tokenizer_2d,
        checkpoint_path=None,  # Load checkpoint after FSDP wrapping
        dtype=config.torch_dtype,
        device=load_device,
    )
    
    # === Apply FSDP Wrapping (if distributed) ===
    if use_fsdp:
        # Get the transformer layer class for wrapping policy
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        
        fsdp_config = DistributedConfig(
            sharding_strategy=config.sharding_strategy,
            mixed_precision="bf16" if config.dtype == "bfloat16" else config.dtype,
            cpu_offload=config.cpu_offload,
            activation_checkpointing=config.gradient_checkpointing,
        )
        
        model = wrap_model_fsdp(
            model,
            fsdp_config,
            transformer_layer_cls=Qwen2DecoderLayer,
            device_id=local_rank,
        )
        print_rank0("Model wrapped with FSDP")
    else:
        # Single GPU: enable gradient checkpointing manually
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
    
    # === Print Training Info ===
    effective_batch_size = config.batch_size * config.grad_accum_steps * world_size
    
    print_rank0(f"\nStarting training...")
    print_rank0(f"  Epochs: {config.epochs}")
    print_rank0(f"  Batch size per GPU: {config.batch_size}")
    print_rank0(f"  Gradient accumulation: {config.grad_accum_steps}")
    print_rank0(f"  World size: {world_size}")
    print_rank0(f"  Effective batch size: {effective_batch_size}")
    print_rank0(f"  Learning rate: {config.lr}")
    print_rank0(f"  FIM ratio: {config.fim_ratio:.0%}")
    print_rank0(f"  Loss on output only: {config.loss_on_output_only}")
    print_rank0()
    
    # === Training Loop ===
    best_eval_loss = float("inf")
    train_loss = float("inf")
    
    try:
        for epoch in range(start_epoch, config.epochs):
            print_rank0(f"\n{'=' * 40}")
            print_rank0(f"Epoch {epoch + 1}/{config.epochs}")
            print_rank0(f"{'=' * 40}")
            
            global_step, train_loss = train_epoch(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                config=config,
                global_step=global_step,
                use_fsdp=use_fsdp,
            )
            
            print_rank0(f"\nEpoch {epoch + 1} complete. Train loss: {train_loss:.4f}")
            
            # === Evaluation ===
            if eval_dataloader:
                eval_loss = evaluate(model, eval_dataloader, device, config, use_fsdp)
                print_rank0(f"Eval loss: {eval_loss:.4f}")
                
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_path = output_dir / f"phase{config.phase}_best.pt"
                    
                    if use_fsdp:
                        save_fsdp_checkpoint(
                            model, optimizer, scheduler,
                            epoch + 1, global_step, eval_loss,
                            str(best_path), rank
                        )
                    else:
                        from src.model.qwen_2d import save_checkpoint
                        save_checkpoint(
                            model, optimizer, scheduler,
                            epoch + 1, global_step, eval_loss,
                            str(best_path), config.phase
                        )
            
            # === Save Epoch Checkpoint ===
            epoch_path = output_dir / f"phase{config.phase}_epoch{epoch + 1}.pt"
            
            if use_fsdp:
                save_fsdp_checkpoint(
                    model, optimizer, scheduler,
                    epoch + 1, global_step, train_loss,
                    str(epoch_path), rank
                )
            else:
                from src.model.qwen_2d import save_checkpoint
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch + 1, global_step, train_loss,
                    str(epoch_path), config.phase
                )
            
            if config.max_steps and global_step >= config.max_steps:
                print_rank0(f"Reached max_steps ({config.max_steps}), stopping.")
                break
        
        # === Final Save ===
        final_path = output_dir / f"phase{config.phase}_final.pt"
        
        if use_fsdp:
            save_fsdp_checkpoint(
                model, optimizer, scheduler,
                config.epochs, global_step, train_loss,
                str(final_path), rank
            )
        else:
            from src.model.qwen_2d import save_checkpoint
            save_checkpoint(
                model, optimizer, scheduler,
                config.epochs, global_step, train_loss,
                str(final_path), config.phase
            )
        
        print_rank0(f"\n{'=' * 60}")
        print_rank0("Training complete!")
        print_rank0(f"Final checkpoint: {final_path}")
        print_rank0(f"{'=' * 60}")
    
    finally:
        # === Cleanup ===
        if use_fsdp:
            cleanup_distributed()


def main():
    """
    Example main function - customize config as needed.
    
    For multi-GPU, run with:
        torchrun --nproc_per_node=8 -m src.train.finetune_distributed
    """
    # === Configure your training here ===
    config = Phase1Config(
        data_dir="/root/arc_data/train_data",
        distributed=True,  # Set to True for multi-GPU
        batch_size=4,
        grad_accum_steps=2,
        lr=3e-5,
    )
    
    train(config)


if __name__ == "__main__":
    main()

