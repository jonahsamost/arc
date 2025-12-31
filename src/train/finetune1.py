"""
ARC-AGI Finetuning Script - Phase 1

Three-phase training architecture:
- Phase 1: Fresh Qwen model, initial finetuning
- Phase 2: Load phase 1 checkpoint, continue with different data
- Phase 3: Load phase 2 checkpoint, final refinement

This script handles all three phases via configuration.

Usage:
    # Phase 1
    python -m src.train.finetune1
    
    # Or import and customize config
    from src.train.finetune1 import train
    from src.train.config import Phase2Config
    
    config = Phase2Config(
        data_dir="/path/to/phase2/data",
        checkpoint="/path/to/phase1.pt",
    )
    train(config)
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from pathlib import Path
from typing import Optional

from src.data.tokenizer_2d import Arc2DTokenizer
from src.data.dataloader import create_dataloader
from src.model.qwen_2d import load_qwen_2d, save_checkpoint, load_checkpoint
from src.train.config import TrainConfig, Phase1Config
from src.train.loss import compute_ntp_loss


def get_optimizer(model, config: TrainConfig):
    """Get optimizer - 8-bit AdamW if configured, else standard AdamW."""
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.95),
            )
            print("Using 8-bit AdamW (bitsandbytes)")
            return optimizer
        except ImportError:
            print("Warning: bitsandbytes not installed, falling back to standard AdamW")
    
    return AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device: torch.device,
    config: TrainConfig,
    global_step: int,
) -> tuple[int, float]:
    """
    Train for one epoch.
    
    Returns:
        (final_step, avg_loss)
    """
    model.train()
    total_loss = 0.0
    step_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    
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
        
        if (batch_idx + 1) % config.grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            total_loss += step_loss * config.grad_accum_steps
            num_batches += config.grad_accum_steps
            
            if global_step % config.log_steps == 0:
                avg_loss = step_loss * config.grad_accum_steps
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "step": global_step,
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.2e}"
                })
            
            step_loss = 0.0
            
            if config.max_steps and global_step >= config.max_steps:
                break
    
    avg_loss = total_loss / max(num_batches, 1)
    return global_step, avg_loss


def evaluate(model, dataloader, device: torch.device, config: TrainConfig) -> float:
    """
    Evaluate model on validation set.
    
    Returns average loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            loss = compute_ntp_loss(
                model, batch, device,
                use_amp=config.use_amp,
                amp_dtype=config.torch_dtype,
                loss_on_output_only=config.loss_on_output_only,
            )
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


def train(config: TrainConfig) -> None:
    """
    Main training function.
    
    Args:
        config: Training configuration dataclass
    """
    # Validate config
    config.validate()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print(f"ARC-AGI Finetuning - Phase {config.phase}")
    print(f"Device: {device}, Dtype: {config.torch_dtype}")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer_2d = Arc2DTokenizer(model_name=config.model_name)
    
    # Load model
    model, tokenizer_2d = load_qwen_2d(
        model_name=config.model_name,
        tokenizer_2d=tokenizer_2d,
        checkpoint_path=None,  # We'll load separately for optimizer state
        dtype=config.torch_dtype,
        device=str(device),
    )
    
    # Enable gradient checkpointing for memory efficiency
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        data_dir=config.data_dir,
        tokenizer=tokenizer_2d,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle_shards=True,
        fim_ratio=config.fim_ratio,  # Mix of NTP and FIM samples
    )
    
    eval_dataloader = None
    if config.eval_dir:
        eval_dataloader = create_dataloader(
            data_dir=config.eval_dir,
            tokenizer=tokenizer_2d,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle_shards=False,
        )
    
    # Setup optimizer (8-bit AdamW if configured)
    optimizer = get_optimizer(model, config)
    
    # Setup scheduler (warmup + cosine decay)
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
    
    # Load checkpoint if resuming
    global_step = 0
    start_epoch = 0
    
    if config.checkpoint:
        metadata = load_checkpoint(config.checkpoint, model, optimizer, scheduler)
        global_step = metadata["step"]
        start_epoch = metadata["epoch"]
        print(f"Resumed from checkpoint: epoch {start_epoch}, step {global_step}")
    
    # Training info
    print(f"\nStarting training...")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.grad_accum_steps}")
    print(f"  Effective batch size: {config.effective_batch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  FIM ratio: {config.fim_ratio:.0%}")
    print(f"  Loss on output only: {config.loss_on_output_only}")
    print()
    
    best_eval_loss = float("inf")
    train_loss = float("inf")
    
    for epoch in range(start_epoch, config.epochs):
        print(f"\n{'=' * 40}")
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"{'=' * 40}")
        
        global_step, train_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            global_step=global_step,
        )
        
        print(f"\nEpoch {epoch + 1} complete. Train loss: {train_loss:.4f}")
        
        # Evaluation
        if eval_dataloader:
            eval_loss = evaluate(model, eval_dataloader, device, config)
            print(f"Eval loss: {eval_loss:.4f}")
            
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_path = output_dir / f"phase{config.phase}_best.pt"
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch + 1, global_step, eval_loss,
                    str(best_path), config.phase
                )
        
        # Save epoch checkpoint
        epoch_path = output_dir / f"phase{config.phase}_epoch{epoch + 1}.pt"
        save_checkpoint(
            model, optimizer, scheduler,
            epoch + 1, global_step, train_loss,
            str(epoch_path), config.phase
        )
        
        if config.max_steps and global_step >= config.max_steps:
            print(f"Reached max_steps ({config.max_steps}), stopping.")
            break
    
    # Final save
    final_path = output_dir / f"phase{config.phase}_final.pt"
    save_checkpoint(
        model, optimizer, scheduler,
        config.epochs, global_step, train_loss,
        str(final_path), config.phase
    )
    
    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"Final checkpoint: {final_path}")
    print(f"{'=' * 60}")


def main():
    config = Phase1Config( data_dir='/root/arc_data/train_data')
    train(config)


# if __name__ == "__main__":
#     main()
