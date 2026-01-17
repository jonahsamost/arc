"""
Evaluation script for ARC-AGI.

Supports two modes:
1. Baseline (1D): Evaluates base Qwen model with simple 1D tokenization
2. Finetuned (2D): Evaluates finetuned checkpoints with 2D RoPE and proper tokenization

The mode is determined by whether a checkpoint is provided.
"""
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from src.data.build_arc_dataset import (
    baseline_eval_all_arc_1d,
    baseline_eval_arc1_1d,
    baseline_eval_arc2_1d,
    baseline_eval_all_arc_2d,
    baseline_eval_arc1_2d,
    baseline_eval_arc2_2d,
)
from src.data.dataloader import ArcDataset, collate_arc_2d
from src.data.tokenizer_1d import ArcBaselineTokenizer, TOK_OUTPUT_SEP, TOK_PAIR_END
from src.data.tokenizer_2d import Arc2DTokenizer


# ANSI color codes for grid visualization
COLORS = {
    0: '\033[40m',   # Black background
    1: '\033[44m',   # Blue
    2: '\033[42m',   # Green
    3: '\033[43m',   # Yellow
    4: '\033[41m',   # Red
    5: '\033[45m',   # Magenta
    6: '\033[46m',   # Cyan
    7: '\033[47m',   # White
    8: '\033[100m',  # Bright black (gray)
    9: '\033[103m',  # Bright yellow (orange-ish)
}
RESET = '\033[0m'


def print_grid(grid: np.ndarray, title: str = "", use_color: bool = True) -> None:
    """
    Print a grid with optional ANSI colors for visualization.
    
    Args:
        grid: 2D numpy array with values 0-9
        title: Optional title to print above grid
        use_color: If True, use ANSI colors; if False, just print numbers
    """
    if grid is None:
        print(f"{title}: None")
        return
    
    if title:
        print(f"{title} ({grid.shape[0]}x{grid.shape[1]}):")
    
    for row in grid:
        line = ""
        for val in row:
            val = int(val)
            if use_color and 0 <= val <= 9:
                # Print colored block with the digit
                line += f"{COLORS[val]} {val} {RESET}"
            else:
                line += f" {val} "
        print(line)
    print()


def print_puzzle(puzzle_data: dict, title: str = "Puzzle", use_color: bool = True) -> None:
    """
    Print a full ARC puzzle with all train and test pairs.
    
    Args:
        puzzle_data: Dict with 'train' and 'test' keys containing input/output pairs
        title: Title for the puzzle
        use_color: Whether to use ANSI colors
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # Print training examples
    train_pairs = puzzle_data.get('train', [])
    for i, pair in enumerate(train_pairs):
        print(f"\n--- Train Example {i+1}/{len(train_pairs)} ---")
        if 'input' in pair:
            inp = np.array(pair['input'])
            print_grid(inp, "Input", use_color)
        if 'output' in pair:
            out = np.array(pair['output'])
            print_grid(out, "Output", use_color)
    
    # Print test examples
    test_pairs = puzzle_data.get('test', [])
    for i, pair in enumerate(test_pairs):
        print(f"\n--- Test Example {i+1}/{len(test_pairs)} ---")
        if 'input' in pair:
            inp = np.array(pair['input'])
            print_grid(inp, "Input", use_color)
        if 'output' in pair:
            out = np.array(pair['output'])
            print_grid(out, "Expected Output", use_color)


def print_test_comparison(
    gt_grid: np.ndarray, 
    pred_grid: Optional[np.ndarray], 
    filename: str,
    use_color: bool = True
) -> None:
    """
    Print ground truth vs predicted output for the test case.
    """
    print(f"\n{'='*60}")
    print(f"TEST RESULT: {filename}")
    print(f"{'='*60}")
    
    print_grid(gt_grid, "Ground Truth Output", use_color)
    
    if pred_grid is not None:
        print_grid(pred_grid, "Model Prediction", use_color)
        
        # Show difference if shapes match
        if gt_grid.shape == pred_grid.shape:
            diff = (gt_grid != pred_grid).astype(int)
            if diff.sum() > 0:
                print(f"Differences ({diff.sum()} pixels wrong):")
                for i, row in enumerate(diff):
                    line = ""
                    for j, is_diff in enumerate(row):
                        if is_diff:
                            line += f"\033[41m X {RESET}"  # Red X for wrong
                        else:
                            line += f"\033[42m ✓ {RESET}"  # Green check for correct
                    print(line)
                print()
            else:
                print("\033[42m ✓ PERFECT MATCH! \033[0m\n")
        else:
            print(f"\033[41m ✗ Shape mismatch: GT {gt_grid.shape} vs Pred {pred_grid.shape} \033[0m\n")
    else:
        print("\033[41m ✗ Prediction: FAILED TO PARSE \033[0m")
    print()


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    checkpoint: Optional[str] = None
    model_name: str = 'Qwen/Qwen3-4B-Thinking-2507'
    eval_set: Literal['arc1', 'arc2', 'all'] = 'arc1'
    device: str = 'cuda'
    max_new_tokens: int = 1024
    # Auto-determined based on checkpoint presence, but can override
    use_2d: Optional[bool] = None  # None = auto (2D if checkpoint, 1D otherwise)


def parse_grid_from_text_1d(text: str) -> Optional[np.ndarray]:
    """
    Parse grid from 1D tokenizer output: "0 1 0\n1 1 1" (space-separated).
    Returns None if parsing fails.
    """
    try:
        rows = text.strip().split('\n')
        grid = []
        for r in rows:
            cols = [int(c) for c in r.strip().split() if c.isdigit()]
            if cols:
                grid.append(cols)
        
        if not grid:
            return None
        
        # Check rectangular
        if len(set(len(row) for row in grid)) > 1:
            return None
            
        return np.array(grid)
    except Exception:
        return None


def parse_grid_from_tokens_2d(
    token_ids: torch.Tensor,
    tokenizer: Arc2DTokenizer,
) -> Optional[np.ndarray]:
    """
    Parse grid from 2D tokenizer output (consecutive digit tokens, newline-separated).
    Returns None if parsing fails.
    """
    try:
        # Create reverse mapping from token ID to digit
        id_to_digit = {tid: i for i, tid in enumerate(tokenizer.digit_ids)}
        # structural_ids now stores lists, get first element for single-token special tokens
        grid_end_ids = tokenizer.structural_ids.get(tokenizer.TOK_GRID_END, [])
        grid_end_id = grid_end_ids[0] if grid_end_ids else None
        newline_id = tokenizer.newline_id
        
        rows = []
        current_row = []
        
        for tid in token_ids.tolist():
            if tid == grid_end_id:
                if current_row:
                    rows.append(current_row)
                break
            elif tid == newline_id:
                if current_row:
                    rows.append(current_row)
                    current_row = []
            elif tid in id_to_digit:
                current_row.append(id_to_digit[tid])
            # Skip other tokens (structural, dimension, etc.)
        
        if not rows:
            return None
        
        # Check rectangular
        if len(set(len(row) for row in rows)) > 1:
            return None
        
        return np.array(rows)
    except Exception as e:
        print(f'Parse grid exception: {e}')
        return None


def calculate_metrics(pred_grid: Optional[np.ndarray], gt_grid: np.ndarray):
    """
    Returns (is_exact_match, pixel_accuracy_float)
    """
    if pred_grid is None:
        return 0, 0.0

    if pred_grid.shape != gt_grid.shape:
        return 0, 0.0
    
    matches = (pred_grid == gt_grid)
    num_correct = np.sum(matches)
    total_pixels = pred_grid.size
    
    pixel_acc = num_correct / total_pixels
    exact_match = 1 if num_correct == total_pixels else 0
    
    return exact_match, pixel_acc


def load_model_1d(model_name: str, device: str = "cuda"):
    """Load base Qwen model for 1D evaluation."""
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype="auto", 
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_model_2d(
    model_name: str,
    checkpoint_path: str,
    tokenizer_2d: Arc2DTokenizer,
    device: str = "cuda",
):
    """
    Load finetuned model with 2D RoPE for evaluation.
    
    This mirrors the training setup:
    1. Load base model
    2. Add special tokens (<|grid_start|>, <|grid_end|>)
    3. Apply 2D RoPE surgery
    4. Load checkpoint weights
    """
    from src.model.qwen_2d import apply_rope_surgery
    
    print(f"Loading model for 2D eval: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load base model to CPU first (for embedding resize)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    )
    
    # Add special tokens and resize embeddings (MUST match training)
    _, model = tokenizer_2d.add_special_tokens_to_model(model)
    
    # Apply 2D RoPE surgery (MUST match training)
    print("Applying 2D RoPE surgery...")
    apply_rope_surgery(model)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Handle torch.compile checkpoints
    was_compiled = any(k.startswith("_orig_mod.") for k in state_dict.keys())
    if was_compiled:
        print("Detected torch.compile checkpoint, applying torch.compile...")
        model = torch.compile(model)
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    
    if "epoch" in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if "step" in checkpoint:
        print(f"  Checkpoint step: {checkpoint['step']}")
    if "loss" in checkpoint:
        print(f"  Checkpoint loss: {checkpoint['loss']:.4f}")
    
    model.eval()
    return model


def evaluate_1d(
    model,
    tokenizer_wrapper: ArcBaselineTokenizer,
    data_path: str,
    device: str = "cuda",
    max_new_tokens: int = 1024,
):
    """
    Run 1D evaluation (for base model).
    Uses simple prompt format with => separator.
    """
    hf_tokenizer = tokenizer_wrapper.tokenizer
    
    # Load raw data to display full puzzles
    arc_dataset_raw = ArcDataset(data_dir=data_path, raw=True)
    
    total_tasks = 0
    total_exact_matches = 0
    total_pixel_acc = 0.0
    
    sep_token_id = tokenizer_wrapper.sep_ids[TOK_OUTPUT_SEP]
    stop_token_id = tokenizer_wrapper.sep_ids[TOK_PAIR_END]
    
    with torch.no_grad():
        for raw_data, filename in tqdm(arc_dataset_raw, desc="Evaluating (1D)"):
            # Get the raw puzzle for visualization
            puzzle_data = raw_data.get('puzzle', raw_data)
            
            # Tokenize for model
            sample = tokenizer_wrapper.build_sample(puzzle_data)
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            labels = sample['labels'].unsqueeze(0).to(device)
            
            # Find the LAST occurrence of <|output|> separator
            sep_indices = (input_ids[0] == sep_token_id).nonzero(as_tuple=True)[0]
            
            if len(sep_indices) == 0:
                print(f"Skipping malformed batch (no separator): {filename}")
                continue
                
            cut_idx = sep_indices[-1].item() + 1
            prompt_ids = input_ids[:, :cut_idx]
            
            # Ground truth from raw puzzle (test output)
            test_pairs = puzzle_data.get('test', [])
            if not test_pairs or 'output' not in test_pairs[-1]:
                print(f"Skipping {filename}: no test output")
                continue
            gt_grid = np.array(test_pairs[-1]['output'])
                
            # Generate
            generated_ids = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=hf_tokenizer.pad_token_id,
                eos_token_id=stop_token_id
            )
            
            pred_ids = generated_ids[0, cut_idx:]
            
            # Decode and parse prediction
            pred_text = hf_tokenizer.decode(pred_ids, skip_special_tokens=True)
            pred_grid = parse_grid_from_text_1d(pred_text)
                
            exact, pixel = calculate_metrics(pred_grid, gt_grid)
            
            total_tasks += 1
            total_exact_matches += exact
            total_pixel_acc += pixel
            
            # Visualize the full puzzle and results
            print_puzzle(puzzle_data, f"Puzzle: {filename}")
            print_test_comparison(gt_grid, pred_grid, filename)
            print(f"  Exact: {exact}, Pixel acc: {pixel:.2%}")
    
    return {
        'total_tasks': total_tasks,
        'exact_matches': total_exact_matches,
        'pixel_accuracy': total_pixel_acc / max(total_tasks, 1),
        'exact_match_rate': total_exact_matches / max(total_tasks, 1),
    }


def evaluate_2d(
    model,
    tokenizer_2d: Arc2DTokenizer,
    data_path: str,
    device: str = "cuda",
    max_new_tokens: int = 1024,
):
    """
    Run 2D evaluation (for finetuned model).
    
    Uses the full 2D format with:
    - <|grid_start|>, <|grid_end|> tokens
    - Dimension encoding (HxW)
    - Proper 2D positional information for generation
    """
    # Load raw data to display full puzzles
    arc_dataset_raw = ArcDataset(data_dir=data_path, raw=True)
    
    total_tasks = 0
    total_exact_matches = 0
    total_pixel_acc = 0.0
    
    # Get stop token - use <|answer|> as EOS for generation
    answer_token_id = tokenizer_2d.get_answer_token_id()
    
    # Debug: verify tokens are set
    print(f"DEBUG: answer_token_id = {answer_token_id}")
    print(f"DEBUG: structural_ids = {tokenizer_2d.structural_ids}")
    
    if answer_token_id is None:
        raise ValueError("answer_token_id is None! add_special_tokens_to_model() was not called on tokenizer")
    
    with torch.no_grad():
        for raw_data, filename in tqdm(arc_dataset_raw, desc="Evaluating (2D)"):
            # Get the raw puzzle for visualization
            puzzle_data = raw_data.get('puzzle', raw_data)
            
            # Tokenize for model (inference mode = stop before test output)
            sample = tokenizer_2d.build_sample(puzzle_data, inference_mode=True)
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            
            # Ground truth from raw puzzle (test output)
            test_pairs = puzzle_data.get('test', [])
            if not test_pairs or 'output' not in test_pairs[-1]:
                print(f"Skipping {filename}: no test output")
                continue
            gt_grid = np.array(test_pairs[-1]['output'])
            
            # Generate
            # Note: For proper 2D generation, we'd need custom generate that passes pos_2d
            # For now, use standard generate (model will use 1D positions for generated tokens)
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer_2d.tokenizer.pad_token_id,
                eos_token_id=answer_token_id,  # Stop at <|answer|> token
            )
            
            # Extract only the generated tokens (after the prompt)
            prompt_len = input_ids.shape[1]
            pred_ids = generated_ids[0, prompt_len:]
            
            # Debug: show what was generated
            print(f"\nDEBUG: prompt_len={prompt_len}, generated_len={generated_ids.shape[1]}, new_tokens={len(pred_ids)}")
            print(f"DEBUG: First 50 pred_ids: {pred_ids[:50].tolist()}")
            print(f"DEBUG: Decoded first 200 chars: {tokenizer_2d.tokenizer.decode(pred_ids[:100])[:200]}")
            
            # Parse predicted grid
            pred_grid = parse_grid_from_tokens_2d(pred_ids, tokenizer_2d)
            
            exact, pixel = calculate_metrics(pred_grid, gt_grid)
            
            total_tasks += 1
            total_exact_matches += exact
            total_pixel_acc += pixel
            
            # Visualize the full puzzle and results
            print_puzzle(puzzle_data, f"Puzzle: {filename}")
            print_test_comparison(gt_grid, pred_grid, filename)
            print(f"  Exact: {exact}, Pixel acc: {pixel:.2%}")
    
    return {
        'total_tasks': total_tasks,
        'exact_matches': total_exact_matches,
        'pixel_accuracy': total_pixel_acc / max(total_tasks, 1),
        'exact_match_rate': total_exact_matches / max(total_tasks, 1),
    }


def run_eval(config: EvalConfig):
    """Run evaluation with the given config."""
    
    # Determine mode: 2D if checkpoint provided, 1D otherwise
    use_2d = config.use_2d
    if use_2d is None:
        use_2d = config.checkpoint is not None
    
    print(f"\n{'='*60}")
    print(f"ARC-AGI Evaluation")
    print(f"Mode: {'2D (finetuned)' if use_2d else '1D (baseline)'}")
    print(f"Eval set: {config.eval_set}")
    print(f"{'='*60}\n")
    
    if use_2d:
        # 2D evaluation for finetuned checkpoints
        tokenizer_2d = Arc2DTokenizer(model_name=config.model_name)
        model = load_model_2d(
            config.model_name,
            config.checkpoint,
            tokenizer_2d,
            config.device,
        )
        
        # Prepare 2D eval data
        print(f"\nPreparing {config.eval_set} evaluation data (2D format)...")
        if config.eval_set == 'arc1':
            data_path = baseline_eval_arc1_2d()
        elif config.eval_set == 'arc2':
            data_path = baseline_eval_arc2_2d()
        else:
            data_path = baseline_eval_all_arc_2d()
        
        # Run evaluation
        print(f"\nStarting 2D evaluation...")
        results = evaluate_2d(
            model=model,
            tokenizer_2d=tokenizer_2d,
            data_path=data_path,
            device=config.device,
            max_new_tokens=config.max_new_tokens,
        )
    else:
        # 1D evaluation for base model
        model = load_model_1d(config.model_name, config.device)
        tokenizer_wrapper = ArcBaselineTokenizer(config.model_name)
        
        # Prepare 1D eval data
        print(f"\nPreparing {config.eval_set} evaluation data (1D format)...")
        if config.eval_set == 'arc1':
            data_path = baseline_eval_arc1_1d()
        elif config.eval_set == 'arc2':
            data_path = baseline_eval_arc2_1d()
        else:
            data_path = baseline_eval_all_arc_1d()
        
        # Run evaluation
        print(f"\nStarting 1D evaluation...")
        results = evaluate_1d(
            model=model,
            tokenizer_wrapper=tokenizer_wrapper,
            data_path=data_path,
            device=config.device,
            max_new_tokens=config.max_new_tokens,
        )
    
    # Print results
    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS ({config.eval_set.upper()})")
    print("=" * 50)
    print(f"Mode:                 {'2D (finetuned)' if use_2d else '1D (baseline)'}")
    print(f"Total Tasks:          {results['total_tasks']}")
    print(f"Exact Match Accuracy: {100 * results['exact_match_rate']:.2f}%")
    print(f"Avg Pixel Accuracy:   {100 * results['pixel_accuracy']:.2f}%")
    print("=" * 50)
    
    if config.checkpoint:
        print(f"\nCheckpoint: {config.checkpoint}")
    else:
        print(f"\nModel: {config.model_name} (base, no finetuning)")
    
    return results


# === Preset Configurations ===

def eval_baseline_arc1():
    """Evaluate base Qwen on ARC-AGI-1."""
    config = EvalConfig(
        checkpoint=None,
        model_name='Qwen/Qwen3-4B-Thinking-2507',
        eval_set='arc1',
        use_2d=False,
    )
    return run_eval(config)


def eval_baseline_arc2():
    """Evaluate base Qwen on ARC-AGI-2."""
    config = EvalConfig(
        checkpoint=None,
        model_name='Qwen/Qwen3-4B-Thinking-2507',
        eval_set='arc2',
        use_2d=False,
    )
    return run_eval(config)


def eval_finetuned_arc1(checkpoint_path: str):
    """Evaluate finetuned model on ARC-AGI-1."""
    config = EvalConfig(
        checkpoint=checkpoint_path,
        model_name='Qwen/Qwen3-4B-Thinking-2507',
        eval_set='arc1',
        use_2d=True,
    )
    return run_eval(config)


def eval_finetuned_arc2(checkpoint_path: str):
    """Evaluate finetuned model on ARC-AGI-2."""
    config = EvalConfig(
        checkpoint=checkpoint_path,
        model_name='Qwen/Qwen3-4B-Thinking-2507',
        eval_set='arc2',
        use_2d=True,
    )
    return run_eval(config)


def eval_finetuned_all(checkpoint_path: str):
    """Evaluate finetuned model on both ARC-AGI-1 and ARC-AGI-2."""
    config = EvalConfig(
        checkpoint=checkpoint_path,
        model_name='Qwen/Qwen3-4B-Thinking-2507',
        eval_set='all',
        use_2d=True,
    )
    return run_eval(config)


def arc1_2d():
    # Example: Evaluate base model on ARC-1
    # eval_baseline_arc1()
    
    # Example: Evaluate finetuned checkpoint
    # eval_finetuned_arc1("/root/checkpoints/phase1/phase1_checkpoint.pt")
    
    # Default: baseline eval
    config = EvalConfig(
        checkpoint='/root/checkpoints/phase3_best.pt',
        eval_set='arc1',
        use_2d=True,
    )
    run_eval(config)


# if __name__ == '__main__':
#     arc1_2d()


_ = '''
ARC-AGI-1 results 12/30/25 (Base Qwen model, 1D)
------------------------------
Final Results on 400 Tasks
Exact Match Accuracy: 1.50%
Avg Pixel Accuracy:   14.10%
------------------------------
'''


_ = '''
ARC-AGI-1 results 01/07/26 (3 finetune phases, 2d)
------------------------------
------------------------------
'''
