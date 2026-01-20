"""
LoRA training script for ARC models.

Simple single-GPU training script for LoRA adapters.
The base model remains frozen; only LoRA parameters are trained.

This produces LoRA weights that serve as the initialization for TTT.

Usage:
    1. Edit LoRATrainConfig in src/lora/config.py with your paths
    2. Run: python -m src.lora.train
"""

import os
import time

# Silence tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from dataclasses import asdict
from typing import Optional

from src.lora.config import LoRAConfig, LoRATrainConfig
from src.lora.lora_model import (
    apply_lora_to_model,
    get_lora_params,
    save_lora,
    freeze_base_model,
    count_lora_params,
)
from src.data.tokenizer_2d import Arc2DTokenizer
from src.data.dataloader import ArcDataset, collate_arc_2d
from src.model.qwen_2d import load_qwen_2d
from src.train.loss import compute_ntp_loss


def create_dataloader(
    data_dir: str,
    tokenizer: Arc2DTokenizer,
    config: LoRATrainConfig,
    is_train: bool = True,
) -> DataLoader:
    """Create DataLoader for LoRA training."""
    dataset = ArcDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        shuffle_shards=is_train,
        max_seq_length=config.max_seq_length,
        fim_ratio=config.fim_ratio if is_train else 0.0,
    )
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_arc_2d,
        pin_memory=config.num_workers > 0,
    )


def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    config: LoRATrainConfig,
    max_samples: Optional[int] = None,
) -> float:
    """Evaluate model on validation set. Returns average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    num_samples = 0
    
    max_samples = max_samples or config.eval_samples
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            loss = compute_ntp_loss(
                model, batch, device,
                use_amp=config.use_amp,
                amp_dtype=config.torch_dtype,
                loss_on_output_only=config.loss_on_output_only,
            )
            total_loss += loss.item()
            num_batches += 1
            num_samples += batch["input_ids"].size(0)
            
            if max_samples and num_samples >= max_samples:
                break
    
    return total_loss / max(num_batches, 1)


def train(config: LoRATrainConfig) -> None:
    """Main LoRA training function."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = Arc2DTokenizer(model_name=config.model_name)
    
    # Load base model
    print(f"Loading base model from {config.base_checkpoint}...")
    model, tokenizer = load_qwen_2d(
        model_name=config.model_name,
        tokenizer_2d=tokenizer,
        checkpoint_path=config.base_checkpoint if config.base_checkpoint else None,
        dtype=config.torch_dtype,
        device="cpu",  # Load to CPU first
    )
    
    # Apply LoRA
    lora_config = LoRAConfig(
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=config.target_modules,
    )
    apply_lora_to_model(model, lora_config)
    
    # Freeze base model
    freeze_base_model(model)
    
    # Move to device
    model = model.to(device)
    
    # Enable gradient checkpointing (trade compute for memory)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Gradient checkpointing enabled")
    
    # Apply torch.compile if enabled
    if config.use_torch_compile:
        print(f"Compiling model with mode={config.compile_mode}...")
        model = torch.compile(model, mode=config.compile_mode)
    
    # Get LoRA parameters (must work after compile)
    lora_param_list = get_lora_params(model)
    if not lora_param_list:
        raise RuntimeError(
            "No LoRA parameters found after torch.compile. "
            "This may indicate an issue with model wrapping. "
            "Try setting use_torch_compile=False to debug."
        )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = count_lora_params(model)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"LoRA params: {lora_params:,}")
    
    # Create dataloaders
    train_dataloader = create_dataloader(config.data_dir, tokenizer, config, is_train=True)
    eval_dataloader = None
    if config.eval_dir:
        eval_dataloader = create_dataloader(config.eval_dir, tokenizer, config, is_train=False)
    
    # Optimizer (only LoRA params)
    optimizer = AdamW(
        lora_param_list,
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Scheduler
    total_steps = config.max_steps if config.max_steps else config.epochs * 10000
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
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
    
    # W&B init
    wandb_run = None
    if config.use_wandb:
        try:
            import wandb
            run_name = config.wandb_run_name or f"lora_r{config.lora_r}_lr{config.lr}"
            wandb_run = wandb.init(
                project=config.wandb_project,
                name=run_name,
                config=asdict(config),
            )
            print(f"W&B initialized: {wandb_run.url}")
        except Exception as e:
            print(f"W&B init failed: {e}")
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting LoRA Training")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Grad accum: {config.grad_accum_steps}")
    print(f"  Effective batch: {config.batch_size * config.grad_accum_steps}")
    print(f"  Learning rate: {config.lr}")
    print(f"  LoRA rank: {config.lora_r}")
    print("=" * 60 + "\n")
    
    global_step = 0
    best_eval_loss = float("inf")
    
    model.train()
    optimizer.zero_grad()
    step_loss = 0.0
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        step_start_time = time.time()
        for batch_idx, batch in enumerate(pbar):
            # Forward pass
            loss = compute_ntp_loss(
                model, batch, device,
                use_amp=config.use_amp,
                amp_dtype=config.torch_dtype,
                loss_on_output_only=config.loss_on_output_only,
            )
            
            # Accumulate the raw loss value for logging (before scaling)
            step_loss += loss.item()
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / config.grad_accum_steps
            scaled_loss.backward()
            
            # Gradient step
            if (batch_idx + 1) % config.grad_accum_steps == 0:
                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(lora_param_list, config.max_grad_norm)
                grad_norm_float = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                # Average loss over the accumulated batches
                avg_loss = step_loss / config.grad_accum_steps
                lr = scheduler.get_last_lr()[0]
                step_time = time.time() - step_start_time if step_start_time else 0
                
                # Logging
                if global_step % config.log_steps == 0:
                    pbar.set_postfix({
                        "step": global_step,
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                    })
                    
                    if wandb_run:
                        import wandb
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/epoch": epoch + 1,
                            "train/grad_norm": grad_norm_float,
                            "train/step_time": step_time,
                        }, step=global_step)
                
                # Evaluation
                if eval_dataloader is not None and global_step % config.eval_steps == 0:
                    eval_loss = evaluate(model, eval_dataloader, device, config)
                    print(f"\n[Step {global_step}] Eval loss: {eval_loss:.4f}")
                    
                    if wandb_run:
                        import wandb
                        wandb.log({"eval/loss": eval_loss}, step=global_step)
                    
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_lora(model, str(output_dir / "lora_best.pt"))
                        if wandb_run:
                            wandb.log({"eval/best_loss": best_eval_loss}, step=global_step)
                    
                    model.train()
                
                # Save checkpoint
                if global_step % config.save_steps == 0:
                    save_lora(model, str(output_dir / f"lora_step_{global_step}.pt"))
                
                step_loss = 0.0
                
                if config.max_steps and global_step >= config.max_steps:
                    break
                step_start_time = time.time()
        
        if config.max_steps and global_step >= config.max_steps:
            break
    
    # Save final
    save_lora(model, str(output_dir / "lora_final.pt"))
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final checkpoint: {output_dir / 'lora_final.pt'}")
    if best_eval_loss < float("inf"):
        print(f"Best eval loss: {best_eval_loss:.4f}")
    print("=" * 60)
    
    if wandb_run:
        import wandb
        wandb.finish()


def main():
    """Entry point for LoRA training.
    
    Edit LoRATrainConfig in src/lora/config.py before running.
    """
    base_path = '/home/ubuntu/arc/arc'
    config = LoRATrainConfig(
        data_dir=f"{base_path}/arc_data/train_data",
        eval_dir=f"{base_path}/arc_data/eval_data",
        base_checkpoint=f"{base_path}/checkpoints/phase1/phase1_checkpoint.pt",
        batch_size=16,
        grad_accum_steps=2,
        gradient_checkpointing=True,
        lr=3.0e-6,
        epochs=8,
        max_steps=3000,
        warmup_steps=300,
        use_torch_compile=False,
        compile_mode='default',
        loss_on_output_only=True,
        eval_samples=500,
        lora_r=16,
    )
    
    # Validate required paths
    if not config.data_dir or not config.base_checkpoint:
        raise ValueError(
            "Please edit LoRATrainConfig.data_dir in src/lora/config.py "
            "to point to your training data directory."
        )
    
    train(config)


if __name__ == "__main__":
    main()
