"""
Phase 3 data generation for ARC-AGI finetuning.

Creates a focused dataset from official ARC puzzles only:
- ARC-AGI-1 training puzzles
- ARC-AGI-2 training puzzles

Each puzzle gets 15 augmentations.
"""
import random
import json
import shutil
from pathlib import Path
from typing import Iterator, Tuple, Any
from tqdm import tqdm

from src.data.build_arc_dataset import (
    get_data_dir, load_arc_puzzles
)
from src.data.common import ARCAugmenter


def estimate_puzzle_tokens(puzzle: dict) -> int:
    """
    Quick estimate of token count for a puzzle without full tokenization.
    """
    total_pixels = 0
    
    for example in puzzle.get('train', []):
        if 'input' in example:
            inp = example['input']
            total_pixels += len(inp) * len(inp[0]) if inp else 0
        if 'output' in example:
            out = example['output']
            total_pixels += len(out) * len(out[0]) if out else 0
    
    for example in puzzle.get('test', []):
        if 'input' in example:
            inp = example['input']
            total_pixels += len(inp) * len(inp[0]) if inp else 0
        if 'output' in example:
            out = example['output']
            total_pixels += len(out) * len(out[0]) if out else 0
    
    return int(total_pixels * 1.2)


def iter_arc_puzzles(
    variant: int,
    num_augs: int = 15,
    max_puzzles: int = None,
) -> Iterator[Tuple[Any, str]]:
    """
    Generator that yields ARC-AGI puzzles with augmentations.
    
    Args:
        variant: 1 for ARC-AGI-1, 2 for ARC-AGI-2
        num_augs: Number of augmentations per puzzle (0 = no augmentation, return original)
        max_puzzles: Maximum number of base puzzles to use (None = all)
    
    Yields:
        (puzzle_dict, puzzle_id) tuples
    """
    arc_aug = ARCAugmenter()
    puzzles = load_arc_puzzles(variant=variant, training=True)
    
    if max_puzzles:
        puzzles = puzzles[:max_puzzles]
    
    for puzzle_data, filepath in puzzles:
        puzzle_id = f"arc{variant}_{Path(filepath).stem}"
        
        if num_augs == 0:
            # No augmentation - return original puzzle
            yield ({'puzzle': puzzle_data}, puzzle_id)
        else:
            # Apply augmentations
            aug_idx = 0
            for aug_puzzle in arc_aug.augment_puzzle(puzzle_data, num_augmentations=num_augs):
                yield (aug_puzzle, f"{puzzle_id}_aug{aug_idx}")
                aug_idx += 1


def generate_phase3_data(
    num_augmentations: int = 15,
    shard_size: int = 1000,
    max_seq_length: int = 1024 * 8,
    output_name: str = 'train_data_phase3',
):
    """
    Generate Phase 3 training data.
    
    Uses only official ARC puzzles:
    - ARC-AGI-1 training set
    - ARC-AGI-2 training set
    
    Each puzzle gets num_augmentations augmentations.
    
    Args:
        num_augmentations: Augmentations per puzzle
        shard_size: Samples per shard file
        max_seq_length: Filter out puzzles > this estimated token count
        output_name: Output directory name
    """
    data_dir = get_data_dir()
    
    print(f"=== Generating Phase 3 Training Data ===")
    print(f"  Sources: ARC-AGI-1 + ARC-AGI-2 (training sets)")
    print(f"  Augmentations per puzzle: {num_augmentations}")
    print(f"  Max sequence length: {max_seq_length:,}")
    print(f"  Shard size: {shard_size}")
    print()
    
    # Setup output directory
    output_path = data_dir / output_name
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    
    # Collect all puzzles
    print("Step 1: Loading and augmenting puzzles...")
    all_puzzles = []
    skipped_count = 0
    
    # ARC-AGI-1
    print("  Loading ARC-AGI-1...")
    arc1_count = 0
    for puzzle, puzzle_id in iter_arc_puzzles(variant=1, num_augs=num_augmentations):
        est_tokens = estimate_puzzle_tokens(puzzle['puzzle'])
        if est_tokens <= max_seq_length:
            all_puzzles.append((puzzle['puzzle'], puzzle_id))
            arc1_count += 1
        else:
            skipped_count += 1
    print(f"    Added {arc1_count} ARC-AGI-1 samples")
    
    # ARC-AGI-2
    print("  Loading ARC-AGI-2...")
    arc2_count = 0
    for puzzle, puzzle_id in iter_arc_puzzles(variant=2, num_augs=num_augmentations):
        est_tokens = estimate_puzzle_tokens(puzzle['puzzle'])
        if est_tokens <= max_seq_length:
            all_puzzles.append((puzzle['puzzle'], puzzle_id))
            arc2_count += 1
        else:
            skipped_count += 1
    print(f"    Added {arc2_count} ARC-AGI-2 samples")
    
    print(f"\n  Total samples: {len(all_puzzles):,}")
    print(f"  Skipped (too long): {skipped_count:,}")
    
    # Shuffle all puzzles for good mixing
    print("\nStep 2: Shuffling and writing shards...")
    random.shuffle(all_puzzles)
    
    # Write shards
    current_shard = 0
    total_written = 0
    
    pbar = tqdm(total=len(all_puzzles), desc="Writing shards")
    
    for i in range(0, len(all_puzzles), shard_size):
        shard_data = all_puzzles[i:i + shard_size]
        shard_path = output_path / f"shard_{current_shard}.jsonl"
        
        with open(shard_path, 'w') as f:
            for puzzle, puzzle_id in shard_data:
                # Wrap puzzle in expected format
                f.write(json.dumps([{'puzzle': puzzle}, puzzle_id]) + '\n')
        
        total_written += len(shard_data)
        pbar.update(len(shard_data))
        current_shard += 1
    
    pbar.close()
    
    print(f"\n=== Done! ===")
    print(f"Output: {output_path}")
    print(f"Total samples: {total_written:,}")
    print(f"Total shards: {current_shard}")
    print(f"  - ARC-AGI-1: {arc1_count:,}")
    print(f"  - ARC-AGI-2: {arc2_count:,}")
    
    return output_path


def main():
    """Generate Phase 3 data with default settings."""
    return generate_phase3_data(
        num_augmentations=15,
        shard_size=1000,
        max_seq_length=1024 * 8,
    )

