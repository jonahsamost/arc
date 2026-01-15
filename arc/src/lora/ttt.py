"""
Test-Time Training (TTT) for ARC puzzle solving.

Adapts LoRA weights on support examples before inference on the query.
Uses ARCAugmenter from src.data.common for data augmentation.
"""

import torch
from torch.optim import AdamW
from typing import Dict, List, Optional
import random

from src.lora.config import TTTConfig
from src.lora.lora_model import (
    get_lora_state_dict,
    load_lora_state_dict,
    get_lora_params,
)
from src.data.common import ARCAugmenter
from src.data.tokenizer_2d import Arc2DTokenizer
from src.data.dataloader import collate_arc_2d
from src.train.loss import compute_ntp_loss


def ttt_adapt(
    model,
    lora_state_dict: Dict[str, torch.Tensor],
    puzzle: dict,
    tokenizer: Arc2DTokenizer,
    config: TTTConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Adapt LoRA weights on support examples using Test-Time Training.
    
    This implements the inner loop of TTT:
    1. Augment support examples
    2. Load initial LoRA weights
    3. Run gradient descent on augmented support set
    4. Return adapted LoRA state dict
    
    NOTE: This function DOES modify the model in place during training.
    The model will have the adapted weights when this function returns.
    
    Args:
        model: The Qwen model with LoRA applied
        lora_state_dict: Initial LoRA weights to start from
        puzzle: Full puzzle dict with 'train' (support) and 'test' (query) keys
        tokenizer: Arc2DTokenizer for tokenizing examples
        config: TTT configuration
        device: Device to run on
    
    Returns:
        Adapted LoRA state dict
    """
    model.train()
    
    # Load initial LoRA weights
    load_lora_state_dict(model, lora_state_dict)
    
    # Get LoRA parameters for optimizer
    lora_params = get_lora_params(model)
    if not lora_params:
        raise ValueError("No LoRA parameters found in model. Did you call apply_lora_to_model()?")
    
    # Create optimizer for LoRA params only
    optimizer = AdamW(lora_params, lr=config.inner_lr)
    
    # Augment support examples
    augmenter = ARCAugmenter()
    
    # Create a support-only puzzle for augmentation
    support_puzzle = {
        'train': puzzle['train'],
        'test': []  # Empty test set for augmentation
    }
    
    # Generate augmented versions
    augmented_list = augmenter.augment_puzzle(
        support_puzzle,
        num_augmentations=config.num_augmentations,
        augment=True,
        jitter=False,
    )
    
    # Tokenize all augmented support examples
    # Use the same format as fine-tuning: all but one pair as context, last pair to predict
    tokenized_samples = []
    tokenization_failures = 0
    
    for aug_idx, aug_data in enumerate(augmented_list):
        aug_puzzle = aug_data['puzzle']
        train_pairs = aug_puzzle['train']
        
        if len(train_pairs) < 2:
            # Need at least 2 pairs to have context + prediction target
            # If only 1 pair, use it as both context and target
            if len(train_pairs) == 1:
                try:
                    sample_puzzle = {
                        'train': [],
                        'test': train_pairs  # Single pair to predict
                    }
                    sample = tokenizer.build_sample(sample_puzzle, inference_mode=False)
                    if sample['input_ids'].shape[0] <= config.max_seq_length:
                        tokenized_samples.append((sample, f"aug_{aug_idx}_pair_0"))
                except Exception as e:
                    tokenization_failures += 1
                    print(f"[TTT] Tokenization failed for aug_{aug_idx}: {e}")
                    continue
        else:
            # Create multiple samples by rotating which pair is the prediction target
            # This gives us more training signal from the same augmentation
            for target_idx in range(len(train_pairs)):
                try:
                    # All pairs except target become context (train)
                    # Target pair becomes the test pair to predict
                    context_pairs = [p for i, p in enumerate(train_pairs) if i != target_idx]
                    target_pair = train_pairs[target_idx]
                    
                    sample_puzzle = {
                        'train': context_pairs,
                        'test': [target_pair]
                    }
                    sample = tokenizer.build_sample(sample_puzzle, inference_mode=False)
                    
                    # Skip if too long
                    if sample['input_ids'].shape[0] <= config.max_seq_length:
                        tokenized_samples.append((sample, f"aug_{aug_idx}_pair_{target_idx}"))
                except Exception as e:
                    tokenization_failures += 1
                    print(f"[TTT] Tokenization failed for aug_{aug_idx}_pair_{target_idx}: {e}")
                    continue
    
    if tokenization_failures > 0:
        print(f"[TTT] {tokenization_failures} tokenization failures, {len(tokenized_samples)} samples created")
    
    if not tokenized_samples:
        print("[TTT] ERROR: No valid tokenized samples for TTT. Returning original LoRA weights.")
        return get_lora_state_dict(model)
    
    print(f"[TTT] Starting adaptation: {len(tokenized_samples)} samples, {config.inner_epochs} epochs, batch_size={config.inner_batch_size}")
    
    # TTT inner loop - iterate through all samples for each epoch
    total_steps = 0
    for epoch in range(config.inner_epochs):
        # Shuffle samples at the start of each epoch
        random.shuffle(tokenized_samples)
        
        epoch_loss = 0.0
        epoch_batches = 0
        
        # Iterate through all samples in batches
        for batch_start in range(0, len(tokenized_samples), config.inner_batch_size):
            batch_end = min(batch_start + config.inner_batch_size, len(tokenized_samples))
            batch_samples = tokenized_samples[batch_start:batch_end]
            
            # Collate batch using shared collate function
            batch = collate_ttt_batch(batch_samples, device)
            
            # Forward pass and loss
            optimizer.zero_grad()
            loss = compute_ntp_loss(
                model,
                batch,
                device,
                use_amp=True,
                amp_dtype=torch.bfloat16,
                loss_on_output_only=config.loss_on_output_only,
            )
            
            # Backward and update
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(lora_params, config.max_grad_norm)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
            total_steps += 1
        
        avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
        print(f"[TTT] Epoch {epoch + 1}/{config.inner_epochs}: avg_loss={avg_epoch_loss:.4f}, steps={epoch_batches}")
    
    print(f"[TTT] Adaptation complete: {total_steps} total steps")
    
    # Extract adapted LoRA weights
    adapted_state_dict = get_lora_state_dict(model)
    
    # Clear CUDA cache to free memory for inference
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return adapted_state_dict


def collate_ttt_batch(
    samples: List[tuple],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of tokenized samples for TTT.
    
    Uses the shared collate_arc_2d function from dataloader.py to avoid
    code duplication. Moves tensors to device after collation.
    
    Args:
        samples: List of (sample_dict, filename) tuples (same format as dataloader)
        device: Target device for tensors
    
    Returns:
        Batched tensors on device
    """
    # Use shared collate function
    batch = collate_arc_2d(samples)
    
    # Move to device (collate_arc_2d returns CPU tensors)
    return {
        "input_ids": batch["input_ids"].to(device),
        "labels": batch["labels"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "pos_1d": batch["pos_1d"].to(device),
        "pos_2d": batch["pos_2d"].to(device),
        "grid_mode": batch["grid_mode"].to(device),
        "output_mask": batch["output_mask"].to(device),
    }


def ttt_adapt_and_load(
    model,
    puzzle: dict,
    tokenizer: Arc2DTokenizer,
    config: TTTConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Convenience function: adapt LoRA and load weights into model.
    
    This modifies the model in place with adapted weights.
    Returns the adapted state dict (useful for saving/comparing).
    """
    # Get current LoRA weights as starting point
    initial_state_dict = get_lora_state_dict(model)
    
    # Adapt
    adapted_state_dict = ttt_adapt(
        model=model,
        lora_state_dict=initial_state_dict,
        puzzle=puzzle,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )
    
    # Load adapted weights into model
    load_lora_state_dict(model, adapted_state_dict)
    
    return adapted_state_dict
