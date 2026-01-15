"""
Test-Time Training (TTT) for ARC puzzle solving.

Adapts LoRA weights on support examples before inference on the query.
Uses ARCAugmenter from src.data.common for data augmentation.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Dict, List, Tuple, Optional, Any
import random
import copy

from src.lora.config import TTTConfig
from src.lora.lora_model import (
    get_lora_state_dict,
    load_lora_state_dict,
    get_lora_params,
)
from src.data.common import ARCAugmenter
from src.data.tokenizer_2d import Arc2DTokenizer
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
    2. Clone LoRA weights
    3. Run gradient descent on augmented support set
    4. Return adapted LoRA state dict
    
    Args:
        model: The Qwen model with LoRA applied
        lora_state_dict: Initial LoRA weights to start from
        puzzle: Full puzzle dict with 'train' (support) and 'test' (query) keys
        tokenizer: Arc2DTokenizer for tokenizing examples
        config: TTT configuration
        device: Device to run on
    
    Returns:
        Adapted LoRA state dict (does not modify model in place)
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
    for aug_data in augmented_list:
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
                        tokenized_samples.append(sample)
                except Exception:
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
                        tokenized_samples.append(sample)
                except Exception:
                    # Skip malformed samples
                    continue
    
    if not tokenized_samples:
        print("Warning: No valid tokenized samples for TTT. Returning original LoRA weights.")
        return get_lora_state_dict(model)
    
    # TTT inner loop
    for step in range(config.inner_steps):
        # Sample a mini-batch
        batch_samples = random.sample(
            tokenized_samples,
            min(config.inner_batch_size, len(tokenized_samples))
        )
        
        # Collate batch
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
        optimizer.step()
    
    # Extract adapted LoRA weights
    adapted_state_dict = get_lora_state_dict(model)
    
    # Clear CUDA cache to free memory for inference
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return adapted_state_dict


def collate_ttt_batch(
    samples: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of tokenized samples for TTT.
    
    Handles variable-length sequences by padding to max length in batch.
    """
    from src.data.tokenizer_2d import IGNORE_INDEX
    
    max_len = max(s["input_ids"].size(0) for s in samples)
    batch_size = len(samples)
    
    # Prepare output tensors
    input_ids = torch.full((batch_size, max_len), 0, dtype=torch.long)
    labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    pos_1d = torch.zeros((batch_size, max_len), dtype=torch.long)
    pos_2d = torch.zeros((batch_size, max_len, 2), dtype=torch.long)
    grid_mode = torch.zeros((batch_size, max_len), dtype=torch.long)
    output_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    
    for i, sample in enumerate(samples):
        seq_len = sample["input_ids"].size(0)
        
        input_ids[i, :seq_len] = sample["input_ids"]
        labels[i, :seq_len] = sample["labels"]
        attention_mask[i, :seq_len] = sample["attention_mask"]
        pos_1d[i, :seq_len] = sample["pos_1d"]
        pos_2d[i, :seq_len] = sample["pos_2d"]
        grid_mode[i, :seq_len] = sample["grid_mode"]
        output_mask[i, :seq_len] = sample["output_mask"]
    
    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "attention_mask": attention_mask.to(device),
        "pos_1d": pos_1d.to(device),
        "pos_2d": pos_2d.to(device),
        "grid_mode": grid_mode.to(device),
        "output_mask": output_mask.to(device),
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
