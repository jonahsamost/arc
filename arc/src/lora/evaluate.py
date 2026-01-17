"""
Evaluation script for LoRA + TTT on ARC datasets.

Provides end-to-end evaluation pipeline:
1. Load model and LoRA weights
2. For each puzzle: run TTT adaptation, generate prediction, compare to ground truth
3. Report accuracy and timing metrics

Usage:
    1. Edit EvalConfig in this file with your paths
    2. Run: python -m src.lora.evaluate
"""

import torch
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np

from src.lora.config import LoRAConfig, TTTConfig, InferenceConfig, LoRATTTConfig, EvalConfig

from src.lora.lora_model import (
    apply_lora_to_model,
    load_lora,
    get_lora_state_dict,
    load_lora_state_dict,
    freeze_base_model,
)
from src.lora.ttt import ttt_adapt
from src.lora.inference import (
    predict_with_verification,
    extract_grid_from_output,
    generate_with_thinking,
    verify_prediction,
)
from src.data.tokenizer_2d import Arc2DTokenizer
from src.model.qwen_2d import load_qwen_2d


@dataclass
class EvalResult:
    """Result for a single puzzle evaluation."""
    puzzle_id: str
    correct: bool
    predicted_grid: Optional[np.ndarray]
    target_grid: np.ndarray
    generated_text: str
    ttt_time_s: float
    inference_time_s: float
    total_time_s: float


@dataclass
class EvalMetrics:
    """Aggregate metrics for evaluation run."""
    num_puzzles: int = 0
    num_correct: int = 0
    total_time_s: float = 0.0
    avg_ttt_time_s: float = 0.0
    avg_inference_time_s: float = 0.0
    results: List[EvalResult] = field(default_factory=list)
    
    @property
    def accuracy(self) -> float:
        return self.num_correct / max(self.num_puzzles, 1)
    
    @property
    def avg_time_per_puzzle_s(self) -> float:
        return self.total_time_s / max(self.num_puzzles, 1)


def load_puzzle_file(file_path: Path, base_puzzle_id: Optional[str] = None) -> Dict[str, dict]:
    """
    Load puzzles from a single file (JSON or JSONL).
    
    Handles various formats:
    - Single puzzle dict with 'train' and 'test' keys
    - Dict of puzzles: {puzzle_id: puzzle_data, ...}
    - List of puzzles
    - JSONL with one puzzle per line (various formats)
    
    Args:
        file_path: Path to the file
        base_puzzle_id: Default puzzle ID to use (defaults to filename stem)
    
    Returns:
        Dict mapping puzzle_id -> puzzle_data
    """
    puzzles = {}
    base_id = base_puzzle_id or file_path.stem
    
    if file_path.suffix == ".jsonl":
        # JSONL file - one puzzle per line
        with open(file_path, 'r') as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                data = json.loads(line)
                
                if isinstance(data, list) and len(data) >= 2:
                    # Format: [puzzle_data, puzzle_id] or [{'puzzle': ...}, puzzle_id]
                    puzzle_data, puzzle_id = data[0], data[1]
                    if 'puzzle' in puzzle_data:
                        puzzles[puzzle_id] = puzzle_data['puzzle']
                    else:
                        puzzles[puzzle_id] = puzzle_data
                elif isinstance(data, dict):
                    if 'train' in data and 'test' in data:
                        # Single puzzle dict
                        puzzle_id = data.get('id', f"{base_id}_{line_idx}")
                        puzzles[puzzle_id] = data
                    elif 'puzzle' in data:
                        # Wrapped puzzle
                        puzzle_id = data.get('id', f"{base_id}_{line_idx}")
                        puzzles[puzzle_id] = data['puzzle']
                    else:
                        # Assume it's a puzzle
                        puzzle_id = data.get('id', f"{base_id}_{line_idx}")
                        puzzles[puzzle_id] = data
    else:
        # JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            if 'train' in data and 'test' in data:
                # Single puzzle
                puzzles[base_id] = data
            elif 'puzzle' in data:
                # Wrapped puzzle
                puzzles[base_id] = data['puzzle']
            else:
                # Dict of puzzles: {puzzle_id: puzzle_data}
                puzzles = data
        elif isinstance(data, list):
            # List of puzzles
            for i, puzzle in enumerate(data):
                if isinstance(puzzle, dict) and 'puzzle' in puzzle:
                    puzzles[f"{base_id}_{i}"] = puzzle['puzzle']
                else:
                    puzzles[f"{base_id}_{i}"] = puzzle
    
    return puzzles


def load_arc_dataset(dataset_path: str) -> Dict[str, dict]:
    """
    Load ARC dataset from directory or file.
    
    Supports:
    - Directory of JSON/JSONL files
    - Single JSON file with dict of puzzles
    - Single JSONL file with puzzles
    
    Returns dict mapping puzzle_id -> puzzle_data
    """
    path = Path(dataset_path)
    puzzles = {}
    
    if path.is_dir():
        # Directory - load all JSON/JSONL files
        for json_file in sorted(path.glob("*.json*")):
            file_puzzles = load_puzzle_file(json_file)
            puzzles.update(file_puzzles)
    else:
        # Single file
        puzzles = load_puzzle_file(path)
    
    return puzzles


def evaluate_single_puzzle(
    model,
    tokenizer: Arc2DTokenizer,
    puzzle_id: str,
    puzzle: dict,
    initial_lora_state: Dict[str, torch.Tensor],
    ttt_config: TTTConfig,
    inference_config: InferenceConfig,
    device: torch.device,
) -> EvalResult:
    """
    Evaluate a single puzzle with TTT and inference.
    
    Returns EvalResult with timing and accuracy information.
    """
    start_time = time.time()
    
    # Get target output (from test pair)
    target_grid = np.array(puzzle['test'][0]['output'])
    
    # === TTT Adaptation ===
    ttt_start = time.time()
    
    # Reset to initial LoRA weights
    load_lora_state_dict(model, initial_lora_state)
    
    # Adapt on support examples
    adapted_state = ttt_adapt(
        model=model,
        lora_state_dict=initial_lora_state,
        puzzle=puzzle,
        tokenizer=tokenizer,
        config=ttt_config,
        device=device,
    )
    
    # Load adapted weights
    load_lora_state_dict(model, adapted_state)
    
    ttt_time = time.time() - ttt_start
    
    # === Inference ===
    inference_start = time.time()
    
    predicted_grid, generated_text = predict_with_verification(
        model=model,
        tokenizer=tokenizer,
        puzzle=puzzle,
        config=inference_config,
        device=device,
    )
    
    inference_time = time.time() - inference_start
    
    # === Check correctness ===
    correct = verify_prediction(predicted_grid, target_grid)
    
    total_time = time.time() - start_time
    
    return EvalResult(
        puzzle_id=puzzle_id,
        correct=correct,
        predicted_grid=predicted_grid,
        target_grid=target_grid,
        generated_text=generated_text,
        ttt_time_s=ttt_time,
        inference_time_s=inference_time,
        total_time_s=total_time,
    )


def evaluate_arc_dataset(
    model,
    tokenizer: Arc2DTokenizer,
    dataset_path: str,
    ttt_config: TTTConfig,
    inference_config: InferenceConfig,
    device: torch.device,
    max_puzzles: Optional[int] = None,
) -> EvalMetrics:
    """
    Evaluate on an ARC dataset.
    
    Args:
        model: Qwen model with LoRA applied
        tokenizer: Arc2DTokenizer
        dataset_path: Path to ARC dataset (directory or file)
        ttt_config: TTT configuration
        inference_config: Inference configuration
        device: Device to run on
        max_puzzles: If set, only evaluate this many puzzles
    
    Returns:
        EvalMetrics with aggregate results
    """
    # Load dataset
    puzzles = load_arc_dataset(dataset_path)
    print(f"Loaded {len(puzzles)} puzzles from {dataset_path}")
    
    if max_puzzles:
        puzzle_ids = list(puzzles.keys())[:max_puzzles]
        puzzles = {k: puzzles[k] for k in puzzle_ids}
        print(f"Evaluating on {len(puzzles)} puzzles (limited by max_puzzles)")
    
    # Get initial LoRA state (to reset between puzzles)
    initial_lora_state = get_lora_state_dict(model)
    
    # Evaluate each puzzle
    metrics = EvalMetrics()
    
    for puzzle_id, puzzle in tqdm(puzzles.items(), desc="Evaluating"):
        try:
            result = evaluate_single_puzzle(
                model=model,
                tokenizer=tokenizer,
                puzzle_id=puzzle_id,
                puzzle=puzzle,
                initial_lora_state=initial_lora_state,
                ttt_config=ttt_config,
                inference_config=inference_config,
                device=device,
            )
            
            metrics.results.append(result)
            metrics.num_puzzles += 1
            metrics.total_time_s += result.total_time_s
            
            if result.correct:
                metrics.num_correct += 1
            
            # Print progress
            print(f"  {puzzle_id}: {'✓' if result.correct else '✗'} "
                  f"(TTT: {result.ttt_time_s:.1f}s, Inf: {result.inference_time_s:.1f}s)")
            
        except Exception as e:
            print(f"  {puzzle_id}: ERROR - {e}")
            continue
    
    # Compute averages
    if metrics.num_puzzles > 0:
        metrics.avg_ttt_time_s = sum(r.ttt_time_s for r in metrics.results) / metrics.num_puzzles
        metrics.avg_inference_time_s = sum(r.inference_time_s for r in metrics.results) / metrics.num_puzzles
    
    return metrics


def print_eval_summary(metrics: EvalMetrics) -> None:
    """Print a summary of evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Puzzles evaluated: {metrics.num_puzzles}")
    print(f"Correct: {metrics.num_correct}")
    print(f"Accuracy: {metrics.accuracy:.2%}")
    print(f"Total time: {metrics.total_time_s:.1f}s")
    print(f"Avg time per puzzle: {metrics.avg_time_per_puzzle_s:.1f}s")
    print(f"  - Avg TTT time: {metrics.avg_ttt_time_s:.1f}s")
    print(f"  - Avg inference time: {metrics.avg_inference_time_s:.1f}s")
    print("=" * 60)


def main():
    """
    Main evaluation entry point.
    
    Edit EvalConfig at the top of this file before running.
    """
    # === EDIT CONFIG HERE ===
    root_dir = '/home/ubuntu/arc/arc'
    config = EvalConfig(
        dataset_path=f"{root_dir}/arc_data/eval_data/arc1/",
        base_checkpoint=f"{root_dir}/checkpoints/phase2_checkpoint.pt",
        lora_checkpoint=f"{root_dir}/checkpoints/lora.pt",
        max_puzzles=None,  # Set to int to limit evaluation
    )
    
    # Validate required paths
    if not config.dataset_path or not config.base_checkpoint:
        raise ValueError(
            "Please edit EvalConfig in src/lora/evaluate.py with your paths."
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = Arc2DTokenizer(model_name=config.model_name)
    
    # Load model
    print(f"Loading model from {config.base_checkpoint}...")
    model, tokenizer = load_qwen_2d(
        model_name=config.model_name,
        tokenizer_2d=tokenizer,
        checkpoint_path=config.base_checkpoint,
        dtype=config.torch_dtype,
        device="cpu",  # Load to CPU first
    )
    
    # Apply LoRA
    lora_config = LoRAConfig(
        r=config.lora_r,
        alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
    )
    apply_lora_to_model(model, lora_config)
    freeze_base_model(model)
    
    # Load LoRA weights if provided
    if config.lora_checkpoint:
        print(f"Loading LoRA weights from {config.lora_checkpoint}...")
        load_lora(model, config.lora_checkpoint)
    
    # Move to device
    model = model.to(device)
    
    # Create TTT and inference configs
    ttt_config = TTTConfig(
        inner_lr=config.ttt_lr,
        inner_epochs=config.ttt_epochs,
        num_augmentations=config.num_augmentations,
        inner_batch_size=config.ttt_batch_size,
    )
    
    inference_config = InferenceConfig(
        num_candidates=config.num_candidates,
        temperature=config.temperature,
        top_p=config.top_p,
        use_thinking=config.use_thinking,
        max_new_tokens=config.max_new_tokens,
    )
    
    # Print config summary
    print("\n" + "=" * 60)
    print("EVALUATION CONFIG")
    print("=" * 60)
    print(f"Dataset: {config.dataset_path}")
    print(f"Base checkpoint: {config.base_checkpoint}")
    print(f"LoRA checkpoint: {config.lora_checkpoint or '(none)'}")
    print(f"TTT: {config.ttt_epochs} epochs, lr={config.ttt_lr}, {config.num_augmentations} augmentations, batch_size={config.ttt_batch_size}")
    print(f"Inference: {config.num_candidates} candidates, temp={config.temperature}, thinking={config.use_thinking}")
    print("=" * 60 + "\n")
    
    # Run evaluation
    metrics = evaluate_arc_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset_path=config.dataset_path,
        ttt_config=ttt_config,
        inference_config=inference_config,
        device=device,
        max_puzzles=config.max_puzzles,
    )
    
    print_eval_summary(metrics)


# if __name__ == "__main__":
#     main()
