#!/usr/bin/env python3
"""
Test suite for collate function and loss computation.

Run with: python -m src.tests.test_collate_and_loss
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM

from src.data.tokenizer_2d import Arc2DTokenizer, IGNORE_INDEX
from src.data.dataloader import collate_arc_2d


def create_test_samples(tok: Arc2DTokenizer, num_samples: int = 3):
    """Create test samples with varying lengths.
    
    Returns list of (sample, filename) tuples as expected by collate_arc_2d.
    """
    puzzles = [
        {
            "train": [{"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]}],
            "test": [{"input": [[0]], "output": [[1]]}]
        },
        {
            "train": [
                {"input": [[1, 2, 3], [4, 5, 6]], "output": [[7, 8, 9], [0, 1, 2]]},
                {"input": [[0, 0], [1, 1]], "output": [[1, 1], [0, 0]]}
            ],
            "test": [{"input": [[2, 2], [3, 3]], "output": [[4, 4], [5, 5]]}]
        },
        {
            "train": [{"input": [[0]], "output": [[9]]}],
            "test": [{"input": [[1]], "output": [[8]]}]
        },
    ]
    
    samples = []
    for i, puzzle in enumerate(puzzles[:num_samples]):
        sample = tok.build_sample(puzzle, inference_mode=False)
        # collate_arc_2d expects (sample, filename) tuples
        samples.append((sample, f"test_puzzle_{i}.json"))
    
    return samples


def test_collate_padding(tok: Arc2DTokenizer) -> bool:
    """Test that collate_arc_2d correctly pads batches."""
    print("\n=== Test 1: Collate Padding ===")
    
    samples = create_test_samples(tok, num_samples=3)
    
    # Get original lengths (samples are (sample_dict, filename) tuples)
    orig_lengths = [len(s[0]["input_ids"]) for s in samples]
    print(f"  Original lengths: {orig_lengths}")
    
    # Collate
    batch = collate_arc_2d(samples)
    
    max_len = max(orig_lengths)
    batch_size = len(samples)
    
    # Check shapes
    checks = [
        (batch["input_ids"].shape == (batch_size, max_len), 
         f"input_ids shape: {batch['input_ids'].shape} (expected ({batch_size}, {max_len}))"),
        (batch["labels"].shape == (batch_size, max_len),
         f"labels shape: {batch['labels'].shape}"),
        (batch["attention_mask"].shape == (batch_size, max_len),
         f"attention_mask shape: {batch['attention_mask'].shape}"),
        (batch["pos_2d"].shape == (batch_size, max_len, 2),
         f"pos_2d shape: {batch['pos_2d'].shape}"),
        (batch["grid_mode"].shape == (batch_size, max_len),
         f"grid_mode shape: {batch['grid_mode'].shape}"),
    ]
    
    all_passed = True
    for condition, description in checks:
        status = "✓" if condition else "✗"
        print(f"  {status} {description}")
        if not condition:
            all_passed = False
    
    return all_passed


def test_collate_attention_mask(tok: Arc2DTokenizer) -> bool:
    """Test that attention_mask is 0 for padded positions."""
    print("\n=== Test 2: Attention Mask for Padding ===")
    
    samples = create_test_samples(tok, num_samples=3)
    orig_lengths = [len(s[0]["input_ids"]) for s in samples]
    
    batch = collate_arc_2d(samples)
    max_len = batch["input_ids"].shape[1]
    
    all_correct = True
    for i, orig_len in enumerate(orig_lengths):
        # First orig_len positions should have attention_mask=1
        real_mask = batch["attention_mask"][i, :orig_len]
        # Remaining positions should have attention_mask=0
        pad_mask = batch["attention_mask"][i, orig_len:]
        
        real_all_ones = torch.all(real_mask == 1).item()
        pad_all_zeros = torch.all(pad_mask == 0).item() if len(pad_mask) > 0 else True
        
        if not real_all_ones or not pad_all_zeros:
            all_correct = False
            print(f"  ✗ Sample {i}: real positions all 1s: {real_all_ones}, pad positions all 0s: {pad_all_zeros}")
    
    status = "✓" if all_correct else "✗"
    print(f"  {status} Attention mask correct for all samples")
    
    return all_correct


def test_collate_labels_padding(tok: Arc2DTokenizer) -> bool:
    """Test that labels have IGNORE_INDEX for padded positions."""
    print("\n=== Test 3: Labels Padding ===")
    
    samples = create_test_samples(tok, num_samples=3)
    orig_lengths = [len(s[0]["input_ids"]) for s in samples]
    
    batch = collate_arc_2d(samples)
    
    all_correct = True
    for i, orig_len in enumerate(orig_lengths):
        # Padded positions should have labels=IGNORE_INDEX
        pad_labels = batch["labels"][i, orig_len:]
        
        if len(pad_labels) > 0:
            pad_all_ignored = torch.all(pad_labels == IGNORE_INDEX).item()
            if not pad_all_ignored:
                all_correct = False
                print(f"  ✗ Sample {i}: padded labels not all IGNORE_INDEX")
    
    status = "✓" if all_correct else "✗"
    print(f"  {status} Padded positions have IGNORE_INDEX labels")
    
    return all_correct


def test_loss_ignores_masked(tok: Arc2DTokenizer) -> bool:
    """Test that cross-entropy loss ignores IGNORE_INDEX positions."""
    print("\n=== Test 4: Loss Ignores Masked Positions ===")
    
    # Create a simple case where we can compute expected loss
    vocab_size = len(tok.tokenizer)
    seq_len = 10
    batch_size = 2
    
    # Random logits
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Labels: some valid, some IGNORE_INDEX
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[:, :5] = IGNORE_INDEX  # First 5 positions ignored
    
    # Compute loss with ignore_index
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=IGNORE_INDEX
    )
    
    # Manually compute loss only on non-ignored positions
    valid_mask = labels != IGNORE_INDEX
    valid_logits = logits[valid_mask]
    valid_labels = labels[valid_mask]
    manual_loss = F.cross_entropy(valid_logits, valid_labels)
    
    # They should match
    loss_matches = torch.isclose(loss, manual_loss, atol=1e-5).item()
    
    print(f"  Loss with ignore_index: {loss.item():.6f}")
    print(f"  Manual loss on valid positions: {manual_loss.item():.6f}")
    
    status = "✓" if loss_matches else "✗"
    print(f"  {status} Loss correctly ignores masked positions")
    
    # Also test with all IGNORE_INDEX - should not crash
    all_ignored_labels = torch.full((batch_size, seq_len), IGNORE_INDEX, dtype=torch.long)
    try:
        all_ignored_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            all_ignored_labels.view(-1),
            ignore_index=IGNORE_INDEX
        )
        # Result should be 0 or nan (depending on PyTorch version)
        handles_all_ignored = True
    except Exception as e:
        handles_all_ignored = False
        print(f"  ✗ Exception with all-ignored labels: {e}")
    
    status = "✓" if handles_all_ignored else "✗"
    print(f"  {status} Handles all-ignored labels without crash")
    
    return loss_matches and handles_all_ignored


def test_shifted_labels(tok: Arc2DTokenizer) -> bool:
    """Test that labels are correctly shifted for autoregressive prediction."""
    print("\n=== Test 5: Shifted Labels (Autoregressive) ===")
    
    # In autoregressive LM, position i predicts token at position i+1
    # But in our setup, labels[i] should equal input_ids[i] where not masked
    # (the loss function handles the shift internally via logits[:, :-1] vs labels[:, 1:])
    
    puzzle = {
        "train": [{"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]}],
        "test": [{"input": [[0]], "output": [[1]]}]
    }
    
    sample = tok.build_sample(puzzle, inference_mode=False)
    
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    
    # Where labels are not IGNORE_INDEX, they should match input_ids
    valid_mask = labels != IGNORE_INDEX
    
    if valid_mask.sum() == 0:
        print("  ✗ No valid labels found")
        return False
    
    matches = (labels[valid_mask] == input_ids[valid_mask]).all().item()
    
    print(f"  Total tokens: {len(input_ids)}")
    print(f"  Valid (non-masked) labels: {valid_mask.sum().item()}")
    
    status = "✓" if matches else "✗"
    print(f"  {status} Labels match input_ids at valid positions")
    
    return matches


def test_loss_decreases_with_correct_predictions(tok: Arc2DTokenizer) -> bool:
    """Test that loss is lower when predictions are correct."""
    print("\n=== Test 6: Loss Decreases with Correct Predictions ===")
    
    vocab_size = len(tok.tokenizer)
    seq_len = 20
    batch_size = 1
    
    # Create labels
    labels = torch.randint(0, 100, (batch_size, seq_len))  # Use low token IDs
    labels[:, :10] = IGNORE_INDEX
    
    # Logits that predict random tokens
    random_logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Logits that predict correct tokens (high probability for correct label)
    correct_logits = torch.randn(batch_size, seq_len, vocab_size) * 0.1
    for i in range(seq_len):
        if labels[0, i] != IGNORE_INDEX:
            correct_logits[0, i, labels[0, i]] = 10.0  # High logit for correct token
    
    random_loss = F.cross_entropy(
        random_logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=IGNORE_INDEX
    )
    
    correct_loss = F.cross_entropy(
        correct_logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=IGNORE_INDEX
    )
    
    loss_decreased = correct_loss < random_loss
    
    print(f"  Random predictions loss: {random_loss.item():.4f}")
    print(f"  Correct predictions loss: {correct_loss.item():.4f}")
    
    status = "✓" if loss_decreased else "✗"
    print(f"  {status} Loss is lower with correct predictions")
    
    return loss_decreased


def run_all_tests():
    """Run all collate and loss tests."""
    print("=" * 60)
    print("Collate and Loss Test Suite")
    print("=" * 60)
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tok = Arc2DTokenizer()
    
    # Load model to add special tokens
    print("Loading model to add special tokens...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B-Thinking-2507",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tok.add_special_tokens_to_model(model)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"Failed to load model: {e}")
        return 1
    
    # Run tests
    results = []
    results.append(("1. Collate Padding", test_collate_padding(tok)))
    results.append(("2. Attention Mask for Padding", test_collate_attention_mask(tok)))
    results.append(("3. Labels Padding", test_collate_labels_padding(tok)))
    results.append(("4. Loss Ignores Masked Positions", test_loss_ignores_masked(tok)))
    results.append(("5. Shifted Labels (Autoregressive)", test_shifted_labels(tok)))
    results.append(("6. Loss Decreases with Correct Predictions", test_loss_decreases_with_correct_predictions(tok)))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        if result:
            status = "PASSED"
        else:
            status = "FAILED"
            all_passed = False
        print(f"  {name}: {status}")
    
    print()
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
