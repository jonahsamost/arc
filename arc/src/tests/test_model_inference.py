#!/usr/bin/env python3
"""
Test model inference to verify Phase 1 finetuning is working correctly.

Loads a checkpoint, runs inference on eval puzzles, and displays:
- Predicted vs target grids (visual comparison)
- Per-pixel accuracy
- Value distribution (detect mode collapse)

Run with: python -m src.tests.test_model_inference --checkpoint <path> --eval_dir <path>
"""

import sys
import argparse
import json
import random
from pathlib import Path
from collections import Counter
from typing import Optional, List, Tuple, Dict
import numpy as np
import torch

from src.data.tokenizer_2d import Arc2DTokenizer
from src.model.qwen_2d import load_qwen_2d


def _set_2d_positions(model, pos_2d: torch.Tensor, grid_mode: torch.Tensor) -> None:
    """
    Store 2D position tensors on attention layers for forward pass access.
    """
    # Handle FSDP wrapping or torch.compile
    base_model = model
    if hasattr(model, '_fsdp_wrapped_module'):
        base_model = model._fsdp_wrapped_module
    elif hasattr(model, 'module'):
        base_model = model.module
    elif hasattr(model, '_orig_mod'):
        base_model = model._orig_mod
    
    # Access layers
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        layers = base_model.model.layers
    else:
        return
    
    for layer in layers:
        attn = layer.self_attn if hasattr(layer, 'self_attn') else layer
        if hasattr(attn, '_fsdp_wrapped_module'):
            attn = attn._fsdp_wrapped_module
        attn._pos_2d = pos_2d
        attn._grid_mode = grid_mode


def load_eval_puzzles(eval_dir: str, max_puzzles: int = 10) -> List[Tuple[dict, str]]:
    """Load evaluation puzzles from directory."""
    eval_path = Path(eval_dir)
    puzzles = []
    
    # Handle both directory of JSON files and JSONL shards
    json_files = list(eval_path.glob("*.json"))
    jsonl_files = list(eval_path.glob("*.jsonl"))
    
    if json_files:
        # Directory of individual JSON puzzle files
        for filepath in json_files[:max_puzzles]:
            with open(filepath) as f:
                puzzle = json.load(f)
            puzzles.append((puzzle, filepath.stem))
    elif jsonl_files:
        # JSONL shard format
        for shard_path in jsonl_files:
            with open(shard_path) as f:
                for line in f:
                    if len(puzzles) >= max_puzzles:
                        break
                    data = json.loads(line)
                    # Handle [puzzle_dict, puzzle_id] or {'puzzle': ..., 'id': ...} formats
                    if isinstance(data, list):
                        puzzle_data, puzzle_id = data[0], data[1]
                        puzzle = puzzle_data.get('puzzle', puzzle_data)
                    else:
                        puzzle = data.get('puzzle', data)
                        puzzle_id = data.get('id', shard_path.stem)
                    puzzles.append((puzzle, puzzle_id))
            if len(puzzles) >= max_puzzles:
                break
    else:
        raise ValueError(f"No .json or .jsonl files found in {eval_dir}")
    
    return puzzles


def grid_to_string(grid: np.ndarray, indent: str = "") -> str:
    """Convert grid to readable string representation."""
    if grid is None:
        return f"{indent}(None)"
    lines = []
    for row in grid:
        lines.append(indent + " ".join(str(int(v)) for v in row))
    return "\n".join(lines)


def compute_cell_accuracy(target: np.ndarray, predicted: np.ndarray) -> Tuple[float, int, int]:
    """Compute cell-level accuracy between grids."""
    if predicted is None:
        return 0.0, 0, target.size
    
    # Handle shape mismatch
    if target.shape != predicted.shape:
        # Compute overlap region
        min_h = min(target.shape[0], predicted.shape[0])
        min_w = min(target.shape[1], predicted.shape[1])
        target_crop = target[:min_h, :min_w]
        pred_crop = predicted[:min_h, :min_w]
        correct = np.sum(target_crop == pred_crop)
        total = target.size  # Penalize for wrong shape
        return correct / total, correct, total
    
    correct = np.sum(target == predicted)
    total = target.size
    return correct / total, correct, total


def get_value_distribution(grid: np.ndarray) -> Dict[int, int]:
    """Get distribution of values in grid."""
    if grid is None:
        return {}
    return dict(Counter(grid.flatten().astype(int)))


def generate_output(
    model,
    tokenizer: Arc2DTokenizer,
    puzzle: dict,
    device: torch.device,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Generate output grid for a puzzle.
    
    Returns (predicted_grid, raw_output_text)
    """
    model.eval()
    
    # Tokenize in inference mode (stops after "Output:" for test pair)
    sample = tokenizer.build_sample(puzzle, inference_mode=True)
    
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
    pos_2d = sample["pos_2d"].unsqueeze(0).to(device)
    grid_mode = sample["grid_mode"].unsqueeze(0).to(device)
    
    # Get special token IDs
    grid_start_id = tokenizer.structural_ids[tokenizer.TOK_GRID_START][0]
    grid_end_id = tokenizer.structural_ids[tokenizer.TOK_GRID_END][0]
    answer_id = tokenizer.structural_ids[tokenizer.TOK_ANSWER][0]
    newline_id = tokenizer.newline_id
    digit_ids = set(tokenizer.digit_ids)
    
    # Stop tokens
    stop_ids = {grid_end_id, answer_id}
    if tokenizer.tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.tokenizer.eos_token_id)
    
    # Convert pos_2d and grid_mode to lists for easy extension during generation
    pos_2d_list = pos_2d[0].tolist()  # List of [y, x] pairs
    grid_mode_list = grid_mode[0].tolist()  # List of 0/1 values
    
    # Set initial 2D positions on model
    _set_2d_positions(model, pos_2d, grid_mode)
    
    generated_ids = []
    current_ids = input_ids
    current_mask = attention_mask
    
    # Track grid state for position encoding during generation
    in_grid = False
    grid_row = 0
    grid_col = 0
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Update positions for current length
            current_pos_2d = torch.tensor([pos_2d_list], dtype=torch.long, device=device)
            current_grid_mode = torch.tensor([grid_mode_list], dtype=torch.long, device=device)
            _set_2d_positions(model, current_pos_2d, current_grid_mode)
            
            outputs = model(
                input_ids=current_ids,
                attention_mask=current_mask,
            )
            
            # Get next token logits
            logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            next_token_id = next_token.item()
            generated_ids.append(next_token_id)
            
            # Check stop condition
            if next_token_id in stop_ids:
                break
            
            # Update grid tracking for position encoding
            if next_token_id == grid_start_id:
                in_grid = True
                grid_row = 0
                grid_col = 0
            elif next_token_id == grid_end_id:
                in_grid = False
            elif in_grid:
                if next_token_id == newline_id:
                    grid_row += 1
                    grid_col = 0
                elif next_token_id in digit_ids:
                    grid_col += 1
            
            # Determine grid_mode and pos_2d for new token
            if in_grid and next_token_id in digit_ids:
                pos_2d_list.append([grid_row, grid_col - 1])
                grid_mode_list.append(1)
            elif in_grid and next_token_id == newline_id:
                # Newlines in grid mode get positioned at end of row
                pos_2d_list.append([grid_row - 1, grid_col])
                grid_mode_list.append(1)
            else:
                pos_2d_list.append([0, 0])
                grid_mode_list.append(0)
            
            # Append to sequences
            current_ids = torch.cat([current_ids, next_token], dim=1)
            current_mask = torch.cat([current_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
    
    # Decode generated text
    generated_text = tokenizer.tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    # Extract grid from generated tokens
    predicted_grid = extract_grid_simple(generated_ids, tokenizer)
    
    return predicted_grid, generated_text


def extract_grid_simple(token_ids: List[int], tokenizer: Arc2DTokenizer) -> Optional[np.ndarray]:
    """Simple grid extraction from token IDs."""
    grid_start_id = tokenizer.structural_ids[tokenizer.TOK_GRID_START][0]
    grid_end_id = tokenizer.structural_ids[tokenizer.TOK_GRID_END][0]
    newline_id = tokenizer.newline_id
    digit_ids = tokenizer.digit_ids
    digit_id_set = set(digit_ids)
    
    # Find grid boundaries
    start_idx = -1
    end_idx = -1
    for i, tok in enumerate(token_ids):
        if tok == grid_start_id:
            start_idx = i
        elif tok == grid_end_id and start_idx != -1:
            end_idx = i
            break
    
    if start_idx == -1 or end_idx == -1:
        return None
    
    # Parse grid tokens
    grid_tokens = token_ids[start_idx + 1:end_idx]
    rows = []
    current_row = []
    
    for tok in grid_tokens:
        if tok == newline_id:
            if current_row:
                rows.append(current_row)
                current_row = []
        elif tok in digit_id_set:
            digit_value = digit_ids.index(tok)
            current_row.append(digit_value)
    
    if current_row:
        rows.append(current_row)
    
    if not rows:
        return None
    
    # Validate rectangular
    row_lens = [len(r) for r in rows]
    if len(set(row_lens)) > 1:
        # Use most common length
        most_common = Counter(row_lens).most_common(1)[0][0]
        rows = [r for r in rows if len(r) == most_common]
    
    if not rows:
        return None
    
    return np.array(rows)


def run_inference_test(
    checkpoint_path: str,
    eval_dir: str,
    num_puzzles: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    device: str = "cuda",
):
    """Run inference test on evaluation puzzles."""
    print("=" * 70)
    print("Model Inference Test")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Eval dir: {eval_dir}")
    print(f"Num puzzles: {num_puzzles}")
    print(f"Device: {device}")
    print()
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = Arc2DTokenizer()
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_qwen_2d(
        model_name="Qwen/Qwen3-4B-Thinking-2507",
        tokenizer_2d=tokenizer,
        checkpoint_path=None,  # We'll load the checkpoint separately
        dtype=torch.bfloat16,
        device="cpu",  # Load to CPU first
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"  Checkpoint step: {checkpoint.get('step', 'unknown')}")
        print(f"  Checkpoint loss: {checkpoint.get('loss', 'unknown')}")
    else:
        state_dict = checkpoint
    
    # Handle torch.compile checkpoints (strip _orig_mod. prefix)
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        print("  Stripping _orig_mod. prefix from checkpoint keys...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Load eval puzzles
    print(f"\nLoading eval puzzles from {eval_dir}...")
    puzzles = load_eval_puzzles(eval_dir, max_puzzles=num_puzzles)
    print(f"Loaded {len(puzzles)} puzzles")
    
    # Run inference
    print("\n" + "=" * 70)
    print("Running Inference")
    print("=" * 70)
    
    total_accuracy = 0.0
    total_correct = 0
    total_cells = 0
    exact_matches = 0
    all_predictions = []
    
    for i, (puzzle, puzzle_id) in enumerate(puzzles):
        print(f"\n{'='*70}")
        print(f"Puzzle {i+1}/{len(puzzles)}: {puzzle_id}")
        print("=" * 70)
        
        # Get target output
        test_output = puzzle.get('test', [{}])[0].get('output', [[]])
        target_grid = np.array(test_output)
        
        # Show train examples summary
        train_pairs = puzzle.get('train', [])
        print(f"Train examples: {len(train_pairs)}")
        for j, pair in enumerate(train_pairs):
            inp = np.array(pair.get('input', [[]]))
            out = np.array(pair.get('output', [[]]))
            print(f"  Pair {j+1}: {inp.shape} -> {out.shape}")
        
        # Show test input
        test_input = puzzle.get('test', [{}])[0].get('input', [[]])
        test_input_grid = np.array(test_input)
        print(f"\nTest Input ({test_input_grid.shape[0]}x{test_input_grid.shape[1]}):")
        print(grid_to_string(test_input_grid, indent="  "))
        
        # Generate prediction
        print(f"\nGenerating output (max {max_new_tokens} tokens, temp={temperature})...")
        try:
            predicted_grid, raw_output = generate_output(
                model, tokenizer, puzzle, torch.device(device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            predicted_grid = None
            raw_output = ""
        
        # Show raw output
        print(f"\nRaw generated text ({len(raw_output)} chars):")
        print(f"  {raw_output[:300]}{'...' if len(raw_output) > 300 else ''}")
        
        # Show grids side by side
        print(f"\nTarget ({target_grid.shape[0]}x{target_grid.shape[1]}):")
        print(grid_to_string(target_grid, indent="  "))
        
        if predicted_grid is not None:
            print(f"\nPredicted ({predicted_grid.shape[0]}x{predicted_grid.shape[1]}):")
            print(grid_to_string(predicted_grid, indent="  "))
        else:
            print("\nPredicted: FAILED TO EXTRACT GRID")
        
        # Compute accuracy
        accuracy, correct, total = compute_cell_accuracy(target_grid, predicted_grid)
        total_accuracy += accuracy
        total_correct += correct
        total_cells += total
        
        is_exact = predicted_grid is not None and np.array_equal(target_grid, predicted_grid)
        if is_exact:
            exact_matches += 1
        
        print(f"\nCell accuracy: {accuracy*100:.1f}% ({correct}/{total})")
        print(f"Exact match: {'YES' if is_exact else 'NO'}")
        
        # Value distribution analysis
        target_dist = get_value_distribution(target_grid)
        pred_dist = get_value_distribution(predicted_grid)
        
        print(f"\nValue distribution:")
        print(f"  Target:    {dict(sorted(target_dist.items()))}")
        print(f"  Predicted: {dict(sorted(pred_dist.items()))}")
        
        # Check for mode collapse (all same value)
        if predicted_grid is not None:
            unique_vals = len(np.unique(predicted_grid))
            if unique_vals == 1:
                print(f"  WARNING: Mode collapse detected! All predictions are {predicted_grid[0,0]}")
            elif unique_vals <= 2:
                print(f"  WARNING: Very low diversity - only {unique_vals} unique values")
        
        all_predictions.append({
            'puzzle_id': puzzle_id,
            'accuracy': accuracy,
            'exact_match': is_exact,
            'target_shape': target_grid.shape,
            'pred_shape': predicted_grid.shape if predicted_grid is not None else None,
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    avg_accuracy = total_accuracy / len(puzzles) if puzzles else 0
    overall_accuracy = total_correct / total_cells if total_cells > 0 else 0
    
    print(f"Puzzles tested: {len(puzzles)}")
    print(f"Exact matches: {exact_matches}/{len(puzzles)} ({100*exact_matches/len(puzzles):.1f}%)")
    print(f"Average cell accuracy: {avg_accuracy*100:.1f}%")
    print(f"Overall cell accuracy: {overall_accuracy*100:.1f}% ({total_correct}/{total_cells})")
    
    # Per-puzzle breakdown
    print("\nPer-puzzle results:")
    for pred in all_predictions:
        status = "✓" if pred['exact_match'] else "✗"
        shape_str = f"{pred['target_shape']} -> {pred['pred_shape']}"
        print(f"  {status} {pred['puzzle_id']}: {pred['accuracy']*100:.1f}% ({shape_str})")
    
    return avg_accuracy, exact_matches, len(puzzles)


def main():
    parser = argparse.ArgumentParser(description="Test model inference on ARC puzzles")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--eval_dir", type=str, required=True, help="Path to eval puzzle directory")
    parser.add_argument("--num_puzzles", type=int, default=5, help="Number of puzzles to test")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    run_inference_test(
        checkpoint_path=args.checkpoint,
        eval_dir=args.eval_dir,
        num_puzzles=args.num_puzzles,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
    )


if __name__ == "__main__":
    main()
