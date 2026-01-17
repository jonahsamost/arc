#!/usr/bin/env python3
"""
Test suite for data generation pipeline.

Run with: python -m src.tests.test_data_generation
"""

import sys
import os
import json
import tempfile
import torch
import numpy as np
from transformers import AutoModelForCausalLM

from src.data.tokenizer_2d import Arc2DTokenizer, IGNORE_INDEX


def create_test_puzzle():
    """Create a simple test puzzle."""
    return {
        "train": [
            {"input": [[1, 2, 3], [4, 5, 6]], "output": [[7, 8, 9], [0, 1, 2]]},
            {"input": [[0, 0], [1, 1]], "output": [[1, 1], [0, 0]]},
        ],
        "test": [
            {"input": [[2, 2, 2], [3, 3, 3]], "output": [[4, 4, 4], [5, 5, 5]]}
        ]
    }


def test_sample_shapes(tok: Arc2DTokenizer) -> bool:
    """Test that build_sample produces tensors with correct shapes."""
    print("\n=== Test 1: Sample Tensor Shapes ===")
    
    puzzle = create_test_puzzle()
    sample = tok.build_sample(puzzle, inference_mode=False)
    
    # All tensors should have same sequence length
    seq_len = sample["input_ids"].shape[0]
    
    checks = [
        (sample["input_ids"].shape == (seq_len,), f"input_ids shape: {sample['input_ids'].shape}"),
        (sample["labels"].shape == (seq_len,), f"labels shape: {sample['labels'].shape}"),
        (sample["attention_mask"].shape == (seq_len,), f"attention_mask shape: {sample['attention_mask'].shape}"),
        (sample["pos_1d"].shape == (seq_len,), f"pos_1d shape: {sample['pos_1d'].shape}"),
        (sample["pos_2d"].shape == (seq_len, 2), f"pos_2d shape: {sample['pos_2d'].shape}"),
        (sample["grid_mode"].shape == (seq_len,), f"grid_mode shape: {sample['grid_mode'].shape}"),
        (sample["output_mask"].shape == (seq_len,), f"output_mask shape: {sample['output_mask'].shape}"),
    ]
    
    all_passed = True
    for condition, description in checks:
        status = "✓" if condition else "✗"
        print(f"  {status} {description}")
        if not condition:
            all_passed = False
    
    print(f"  Sequence length: {seq_len}")
    return all_passed


def test_labels_masking(tok: Arc2DTokenizer) -> bool:
    """Test that labels correctly mask input tokens."""
    print("\n=== Test 2: Labels Masking ===")
    
    puzzle = create_test_puzzle()
    sample = tok.build_sample(puzzle, inference_mode=False)
    
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    output_mask = sample["output_mask"]
    
    # Count masked vs unmasked labels
    masked_count = (labels == IGNORE_INDEX).sum().item()
    unmasked_count = (labels != IGNORE_INDEX).sum().item()
    total = len(labels)
    
    # At minimum, some tokens should be masked (input context)
    has_masked = masked_count > 0
    # At minimum, some tokens should be unmasked (output tokens to predict)
    has_unmasked = unmasked_count > 0
    
    # Where labels != IGNORE_INDEX, labels should equal input_ids (autoregressive)
    unmasked_match = torch.all(labels[labels != IGNORE_INDEX] == input_ids[labels != IGNORE_INDEX]).item()
    
    # output_mask should be 1 only for actual grid output pixels
    output_mask_sum = output_mask.sum().item()
    
    print(f"  Total tokens: {total}")
    print(f"  Masked (IGNORE_INDEX): {masked_count} ({100*masked_count/total:.1f}%)")
    print(f"  Unmasked (predicted): {unmasked_count} ({100*unmasked_count/total:.1f}%)")
    print(f"  Output mask sum: {output_mask_sum}")
    
    checks = [
        (has_masked, "Has masked tokens (input context)"),
        (has_unmasked, "Has unmasked tokens (predictions)"),
        (unmasked_match, "Unmasked labels match input_ids"),
        (output_mask_sum > 0, "Output mask has positive entries"),
    ]
    
    all_passed = True
    for condition, description in checks:
        status = "✓" if condition else "✗"
        print(f"  {status} {description}")
        if not condition:
            all_passed = False
    
    return all_passed


def test_grid_mode_alignment(tok: Arc2DTokenizer) -> bool:
    """Test that grid_mode correctly identifies grid pixels and row separators."""
    print("\n=== Test 3: Grid Mode Alignment ===")
    
    puzzle = {
        "train": [{"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]}],
        "test": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}]
    }
    
    sample = tok.build_sample(puzzle, inference_mode=False)
    
    input_ids = sample["input_ids"]
    grid_mode = sample["grid_mode"]
    
    digit_id_set = set(tok.digit_ids)
    newline_id = tok.newline_id
    
    # Check: grid_mode=1 should only be for digit tokens OR newlines (row separators within grids)
    # Newlines within grids have grid_mode=1 for spatial continuity (intentional design)
    errors = []
    for i, (tid, gm) in enumerate(zip(input_ids.tolist(), grid_mode.tolist())):
        if gm == 1:
            # Valid grid_mode=1 tokens: digits (0-9) or newlines (row separators)
            if tid not in digit_id_set and tid != newline_id:
                decoded = tok.tokenizer.decode([tid])
                errors.append(f"Position {i}: grid_mode=1 for unexpected token {tid} ('{decoded}')")
    
    # Check: some tokens should have grid_mode=1
    has_grid_mode_1 = (grid_mode == 1).sum().item() > 0
    
    # Check: some tokens should have grid_mode=0
    has_grid_mode_0 = (grid_mode == 0).sum().item() > 0
    
    # Count digits and newlines with grid_mode=1
    grid_mode_1_tokens = [(tid, gm) for tid, gm in zip(input_ids.tolist(), grid_mode.tolist()) if gm == 1]
    digit_count = sum(1 for tid, _ in grid_mode_1_tokens if tid in digit_id_set)
    newline_count = sum(1 for tid, _ in grid_mode_1_tokens if tid == newline_id)
    
    print(f"  grid_mode=0 count: {(grid_mode == 0).sum().item()}")
    print(f"  grid_mode=1 count: {(grid_mode == 1).sum().item()} (digits: {digit_count}, newlines: {newline_count})")
    
    checks = [
        (len(errors) == 0, f"grid_mode=1 only for digits/newlines ({len(errors)} errors)"),
        (has_grid_mode_1, "Has grid_mode=1 tokens"),
        (has_grid_mode_0, "Has grid_mode=0 tokens"),
        (digit_count > 0, "Has digit tokens with grid_mode=1"),
    ]
    
    all_passed = True
    for condition, description in checks:
        status = "✓" if condition else "✗"
        print(f"  {status} {description}")
        if not condition:
            all_passed = False
    
    if errors:
        print("  Errors:")
        for err in errors[:5]:  # Show first 5
            print(f"    {err}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")
    
    return all_passed


def test_pos_2d_values(tok: Arc2DTokenizer) -> bool:
    """Test that pos_2d contains valid grid coordinates."""
    print("\n=== Test 4: 2D Position Values ===")
    
    puzzle = {
        "train": [{"input": [[1, 2, 3], [4, 5, 6]], "output": [[7, 8], [9, 0]]}],
        "test": [{"input": [[0]], "output": [[1]]}]
    }
    
    sample = tok.build_sample(puzzle, inference_mode=False)
    
    pos_2d = sample["pos_2d"]
    grid_mode = sample["grid_mode"]
    
    # For text tokens (grid_mode=0), pos_2d should be (0, 0)
    text_positions = pos_2d[grid_mode == 0]
    text_all_zero = torch.all(text_positions == 0).item()
    
    # For grid tokens (grid_mode=1), pos_2d should have valid coordinates
    grid_positions = pos_2d[grid_mode == 1]
    
    # Check that coordinates are non-negative
    coords_non_negative = torch.all(grid_positions >= 0).item() if len(grid_positions) > 0 else True
    
    # Check that we see varied coordinates (not all same)
    if len(grid_positions) > 1:
        has_varied_coords = not torch.all(grid_positions == grid_positions[0]).item()
    else:
        has_varied_coords = True
    
    print(f"  Text tokens with pos_2d: {len(text_positions)}")
    print(f"  Grid tokens with pos_2d: {len(grid_positions)}")
    if len(grid_positions) > 0:
        print(f"  Grid pos_2d range: y=[{grid_positions[:, 0].min().item()}, {grid_positions[:, 0].max().item()}], "
              f"x=[{grid_positions[:, 1].min().item()}, {grid_positions[:, 1].max().item()}]")
    
    checks = [
        (text_all_zero, "Text tokens have pos_2d=(0,0)"),
        (coords_non_negative, "Grid coordinates are non-negative"),
        (has_varied_coords, "Grid has varied coordinates"),
    ]
    
    all_passed = True
    for condition, description in checks:
        status = "✓" if condition else "✗"
        print(f"  {status} {description}")
        if not condition:
            all_passed = False
    
    return all_passed


def test_inference_mode_truncation(tok: Arc2DTokenizer) -> bool:
    """Test that inference mode correctly truncates at the right point."""
    print("\n=== Test 5: Inference Mode Truncation ===")
    
    puzzle = create_test_puzzle()
    
    full_sample = tok.build_sample(puzzle, inference_mode=False)
    inf_sample = tok.build_sample(puzzle, inference_mode=True)
    
    full_len = len(full_sample["input_ids"])
    inf_len = len(inf_sample["input_ids"])
    
    # Inference mode should be shorter
    is_shorter = inf_len < full_len
    
    # Decode both and check
    full_decoded = tok.tokenizer.decode(full_sample["input_ids"].tolist())
    inf_decoded = tok.tokenizer.decode(inf_sample["input_ids"].tolist())
    
    # Inference should end with "Output:"
    ends_with_output = inf_decoded.rstrip().endswith("Output:")
    
    # Inference should NOT contain <|answer|>
    no_answer = "<|answer|>" not in inf_decoded
    
    # Full should contain <|answer|>
    has_answer = "<|answer|>" in full_decoded
    
    print(f"  Full sample length: {full_len}")
    print(f"  Inference sample length: {inf_len}")
    
    checks = [
        (is_shorter, "Inference mode produces shorter sequence"),
        (ends_with_output, "Inference mode ends with 'Output:'"),
        (no_answer, "Inference mode does not contain '<|answer|>'"),
        (has_answer, "Full mode contains '<|answer|>'"),
    ]
    
    all_passed = True
    for condition, description in checks:
        status = "✓" if condition else "✗"
        print(f"  {status} {description}")
        if not condition:
            all_passed = False
    
    return all_passed


def test_augmentation(tok: Arc2DTokenizer) -> bool:
    """Test that augmentation produces valid variants."""
    print("\n=== Test 6: Data Augmentation ===")
    
    try:
        from src.data.common import ARCAugmenter
    except ImportError:
        print("  ✗ Could not import ARCAugmenter")
        return False
    
    puzzle = {
        "train": [{"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]}],
        "test": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}]
    }
    
    augmenter = ARCAugmenter()
    
    # Generate some augmented versions
    # augment_puzzle returns a list of dicts with 'aug_id', 'puzzle', 'og_dims' keys
    augmented = augmenter.augment_puzzle(puzzle, num_augmentations=10)
    
    # Check that we get varied results
    unique_inputs = set()
    for aug_item in augmented:
        # The actual puzzle is in aug_item['puzzle']
        aug_puzzle = aug_item['puzzle']
        # Convert to tuple for hashing
        inp = tuple(tuple(row) for row in aug_puzzle["train"][0]["input"])
        unique_inputs.add(inp)
    
    has_variety = len(unique_inputs) > 1
    
    # Check that all augmented puzzles can be tokenized
    all_tokenizable = True
    for aug_item in augmented:
        aug_puzzle = aug_item['puzzle']
        try:
            sample = tok.build_sample(aug_puzzle, inference_mode=False)
            if sample["input_ids"] is None:
                all_tokenizable = False
        except Exception as e:
            all_tokenizable = False
            print(f"  ✗ Tokenization failed: {e}")
    
    print(f"  Generated {len(augmented)} augmented puzzles")
    print(f"  Unique input variations: {len(unique_inputs)}")
    
    checks = [
        (has_variety, "Augmentation produces varied results"),
        (all_tokenizable, "All augmented puzzles are tokenizable"),
    ]
    
    all_passed = True
    for condition, description in checks:
        status = "✓" if condition else "✗"
        print(f"  {status} {description}")
        if not condition:
            all_passed = False
    
    return all_passed


def test_fim_sample(tok: Arc2DTokenizer) -> bool:
    """Test that FIM (Fill-in-the-Middle) samples are built correctly."""
    print("\n=== Test 7: FIM Sample ===")
    
    puzzle = {
        "train": [
            {"input": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]], 
             "output": [[3, 4, 5, 6], [7, 8, 9, 0], [1, 2, 3, 4]]}
        ],
        "test": [
            {"input": [[0, 1, 2, 3], [4, 5, 6, 7]], 
             "output": [[8, 9, 0, 1], [2, 3, 4, 5]]}
        ]
    }
    
    try:
        sample = tok.build_fim_sample(puzzle, target_mask_ratio=0.2)
    except Exception as e:
        print(f"  ✗ build_fim_sample raised exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check that sample has expected keys
    expected_keys = ["input_ids", "labels", "attention_mask", "pos_1d", "pos_2d", "grid_mode", "output_mask"]
    has_all_keys = all(k in sample for k in expected_keys)
    
    if not has_all_keys:
        missing = [k for k in expected_keys if k not in sample]
        print(f"  ✗ Missing keys: {missing}")
        return False
    
    # Decode and check structure
    decoded = tok.tokenizer.decode(sample["input_ids"].tolist())
    
    # Check for FIM tokens
    has_fim_prefix = "<|fim_prefix|>" in decoded
    has_fim_suffix = "<|fim_suffix|>" in decoded
    has_fim_middle = "<|fim_middle|>" in decoded
    has_fim_hole = "<|fim_hole|>" in decoded
    has_answer = "<|answer|>" in decoded
    
    print(f"  Decoded sequence (first 200 chars):\n    {decoded[:200]}...")
    print(f"  Decoded sequence (last 100 chars):\n    ...{decoded[-100:]}")
    
    # Check sequence structure: prefix -> suffix -> middle -> answer
    if has_fim_prefix and has_fim_suffix and has_fim_middle:
        prefix_pos = decoded.find("<|fim_prefix|>")
        suffix_pos = decoded.find("<|fim_suffix|>")
        middle_pos = decoded.find("<|fim_middle|>")
        answer_pos = decoded.find("<|answer|>") if has_answer else len(decoded)
        
        correct_order = prefix_pos < suffix_pos < middle_pos < answer_pos
    else:
        correct_order = False
    
    # Check that labels are mostly IGNORE_INDEX for prefix, real for middle/answer
    labels = sample["labels"]
    input_ids = sample["input_ids"]
    
    masked_count = (labels == IGNORE_INDEX).sum().item()
    unmasked_count = (labels != IGNORE_INDEX).sum().item()
    
    # There should be some unmasked labels (the middle content to predict)
    has_predictions = unmasked_count > 0
    
    print(f"  Total tokens: {len(labels)}")
    print(f"  Masked (IGNORE_INDEX): {masked_count}")
    print(f"  Unmasked (predicted): {unmasked_count}")
    
    checks = [
        (has_all_keys, "Has all expected keys"),
        (has_fim_prefix, "Contains <|fim_prefix|>"),
        (has_fim_suffix, "Contains <|fim_suffix|>"),
        (has_fim_middle, "Contains <|fim_middle|>"),
        (has_fim_hole, "Contains <|fim_hole|> (masked content markers)"),
        (has_answer, "Contains <|answer|>"),
        (correct_order, "FIM tokens in correct order (prefix < suffix < middle < answer)"),
        (has_predictions, "Has tokens to predict (unmasked labels)"),
    ]
    
    all_passed = True
    for condition, description in checks:
        status = "✓" if condition else "✗"
        print(f"  {status} {description}")
        if not condition:
            all_passed = False
    
    return all_passed


def run_all_tests():
    """Run all data generation tests."""
    print("=" * 60)
    print("Data Generation Test Suite")
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
        del model  # Free memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Cannot run tests without model for special tokens.")
        return 1
    
    # Run tests
    results = []
    results.append(("1. Sample Tensor Shapes", test_sample_shapes(tok)))
    results.append(("2. Labels Masking", test_labels_masking(tok)))
    results.append(("3. Grid Mode Alignment", test_grid_mode_alignment(tok)))
    results.append(("4. 2D Position Values", test_pos_2d_values(tok)))
    results.append(("5. Inference Mode Truncation", test_inference_mode_truncation(tok)))
    results.append(("6. Data Augmentation", test_augmentation(tok)))
    results.append(("7. FIM Sample", test_fim_sample(tok)))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        if result is None:
            status = "SKIPPED"
        elif result:
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
