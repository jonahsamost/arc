"""
Evaluation script for LoRA + TTT on ARC datasets.

Provides end-to-end evaluation pipeline:
1. Load model and LoRA weights
2. For each puzzle: run TTT adaptation, generate prediction, compare to ground truth
3. Report accuracy and timing metrics
"""

import torch
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np

from src.lora.config import LoRAConfig, TTTConfig, InferenceConfig, LoRATTTConfig
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


def load_arc_dataset(dataset_path: str) -> Dict[str, dict]:
    """
    Load ARC dataset from directory or file.
    
    Supports:
    - Directory of JSON files (one puzzle per file)
    - Single JSON file with dict of puzzles
    - Single JSONL file with puzzles
    
    Returns dict mapping puzzle_id -> puzzle_data
    """
    path = Path(dataset_path)
    puzzles = {}
    
    if path.is_dir():
        # Directory of JSON files
        for json_file in sorted(path.glob("*.json")):
            puzzle_id = json_file.stem
            with open(json_file, 'r') as f:
                puzzles[puzzle_id] = json.load(f)
    elif path.suffix == ".jsonl":
        # JSONL file
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if isinstance(data, list) and len(data) >= 2:
                        puzzle_data, puzzle_id = data[0], data[1]
                        if 'puzzle' in puzzle_data:
                            puzzles[puzzle_id] = puzzle_data['puzzle']
                        else:
                            puzzles[puzzle_id] = puzzle_data
                    elif isinstance(data, dict):
                        puzzle_id = data.get('id', f"puzzle_{len(puzzles)}")
                        puzzles[puzzle_id] = data
    elif path.suffix == ".json":
        # Single JSON file with dict of puzzles
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                # Could be {puzzle_id: puzzle_data} or a single puzzle
                if 'train' in data and 'test' in data:
                    puzzles[path.stem] = data
                else:
                    puzzles = data
            elif isinstance(data, list):
                for i, puzzle in enumerate(data):
                    puzzles[f"puzzle_{i}"] = puzzle
    
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
    
    Example usage:
        python -m src.lora.evaluate \
            --dataset /path/to/arc_evaluation \
            --checkpoint /path/to/phase3_final.pt \
            --lora_checkpoint /path/to/lora_trained.pt
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate LoRA + TTT on ARC dataset")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to ARC dataset (directory or file)")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to base model checkpoint")
    parser.add_argument("--lora_checkpoint", type=str, default="",
                       help="Path to trained LoRA weights (optional)")
    parser.add_argument("--max_puzzles", type=int, default=None,
                       help="Maximum puzzles to evaluate")
    
    # TTT config
    parser.add_argument("--ttt_lr", type=float, default=1e-3)
    parser.add_argument("--ttt_steps", type=int, default=5)
    parser.add_argument("--num_augmentations", type=int, default=50)
    
    # Inference config
    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--use_thinking", action="store_true", default=True)
    parser.add_argument("--no_thinking", action="store_true")
    
    # LoRA config
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = Arc2DTokenizer()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, tokenizer = load_qwen_2d(
        tokenizer_2d=tokenizer,
        checkpoint_path=args.checkpoint,
        device="cpu",  # Load to CPU first
    )
    
    # Apply LoRA
    lora_config = LoRAConfig(r=args.lora_r, alpha=args.lora_alpha)
    apply_lora_to_model(model, lora_config)
    freeze_base_model(model)
    
    # Load LoRA weights if provided
    if args.lora_checkpoint:
        load_lora(model, args.lora_checkpoint)
    
    # Move to device
    model = model.to(device)
    
    # Create configs
    ttt_config = TTTConfig(
        inner_lr=args.ttt_lr,
        inner_steps=args.ttt_steps,
        num_augmentations=args.num_augmentations,
    )
    
    inference_config = InferenceConfig(
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        use_thinking=args.use_thinking and not args.no_thinking,
    )
    
    # Run evaluation
    metrics = evaluate_arc_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset_path=args.dataset,
        ttt_config=ttt_config,
        inference_config=inference_config,
        device=device,
        max_puzzles=args.max_puzzles,
    )
    
    print_eval_summary(metrics)


if __name__ == "__main__":
    main()
