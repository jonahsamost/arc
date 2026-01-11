"""
Phase 2 data generation for ARC-AGI finetuning.

Creates a combined dataset from:
- Mini-ARC puzzles
- Concept-ARC puzzles  
- RE-ARC puzzles (100 base puzzles)
- ARC-AGI-1 training puzzles
- ARC-AGI-2 training puzzles

Each puzzle gets augmentations (configurable per source).
"""
import random
import json
import shutil
from pathlib import Path
from typing import Iterator, Tuple, Any, Optional
from tqdm import tqdm

from src.data.build_arc_dataset import (
    get_data_dir, get_rearc_path, generate_rearc_data,
    load_mini_arc, load_concept_arc, load_arc_puzzles, sample_puzzle_size
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


def iter_rearc_puzzles_limited(
    num_base_puzzles: int = 100,
    num_augs: int = 10,
) -> Iterator[Tuple[Any, str]]:
    """
    Generator that yields a limited number of RE-ARC puzzles with augmentations.
    
    Args:
        num_base_puzzles: Number of base puzzles to create (before augmentation)
        num_augs: Number of augmentations per puzzle
    
    Yields:
        (puzzle_dict, puzzle_id) tuples
    """
    re_arc_path = get_rearc_path()
    arc_aug = ARCAugmenter()
    path = re_arc_path / 'tasks'
    
    base_puzzles_created = 0
    total_yielded = 0
    
    task_files = list(path.rglob('*.json'))
    random.shuffle(task_files)
    
    for filepath in task_files:
        if base_puzzles_created >= num_base_puzzles:
            return
            
        if filepath.is_file():
            with open(filepath, 'r') as fd:
                all_examples = json.loads(fd.read())
            
            if len(all_examples) < 3:
                continue
            
            random.shuffle(all_examples)
            
            idx = 0
            puzzle_num = 0
            
            while idx < len(all_examples) and base_puzzles_created < num_base_puzzles:
                num_train, num_test = sample_puzzle_size()
                total_needed = num_train + num_test
                
                if idx + total_needed > len(all_examples):
                    remaining = len(all_examples) - idx
                    if remaining >= 3:
                        num_test = min(num_test, remaining // 3)
                        num_train = remaining - num_test
                    else:
                        break
                
                train_examples = all_examples[idx:idx + num_train]
                test_examples = all_examples[idx + num_train:idx + num_train + num_test]
                
                task = {"train": train_examples, "test": test_examples}
                puzzle_id = f"rearc_{filepath.stem}_p{puzzle_num}"
                
                # Apply augmentations and yield each
                aug_idx = 0
                for aug_puzzle in arc_aug.augment_puzzle(task, num_augmentations=num_augs):
                    yield (aug_puzzle, f"{puzzle_id}_aug{aug_idx}")
                    aug_idx += 1
                    total_yielded += 1
                
                base_puzzles_created += 1
                idx += num_train + num_test
                puzzle_num += 1


def iter_mini_arc_puzzles(
    num_augs: int = 10,
    max_puzzles: int = None,
) -> Iterator[Tuple[Any, str]]:
    """
    Generator that yields Mini-ARC puzzles with augmentations.
    """
    arc_aug = ARCAugmenter()
    puzzles = load_mini_arc()
    
    if max_puzzles:
        puzzles = puzzles[:max_puzzles]
    
    for puzzle_data, filepath in puzzles:
        puzzle_id = f"miniarc_{Path(filepath).stem}"
        
        aug_idx = 0
        for aug_puzzle in arc_aug.augment_puzzle(puzzle_data, num_augmentations=num_augs):
            yield (aug_puzzle, f"{puzzle_id}_aug{aug_idx}")
            aug_idx += 1


def iter_concept_arc_puzzles(
    num_augs: int = 10,
    max_puzzles: int = None,
) -> Iterator[Tuple[Any, str]]:
    """
    Generator that yields Concept-ARC puzzles with augmentations.
    """
    arc_aug = ARCAugmenter()
    puzzles = load_concept_arc()
    
    if max_puzzles:
        puzzles = puzzles[:max_puzzles]
    
    for puzzle_data, filepath in puzzles:
        puzzle_id = f"conceptarc_{Path(filepath).stem}"
        
        aug_idx = 0
        for aug_puzzle in arc_aug.augment_puzzle(puzzle_data, num_augmentations=num_augs):
            yield (aug_puzzle, f"{puzzle_id}_aug{aug_idx}")
            aug_idx += 1


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


def generate_phase2_data(
    num_rearc_base_puzzles: int = 100,
    num_augmentations: int = 10,
    arc_augmentations: int = 15,
    shard_size: int = 1000,
    max_seq_length: int = 1024 * 8,
    output_name: str = 'train_data_phase2',
):
    """
    Generate Phase 2 training data.
    
    Combines:
    - Mini-ARC puzzles (all available)
    - Concept-ARC puzzles (all available)
    - RE-ARC puzzles (limited to num_rearc_base_puzzles)
    - ARC-AGI-1 training puzzles
    - ARC-AGI-2 training puzzles
    
    Args:
        num_rearc_base_puzzles: Number of RE-ARC base puzzles to use
        num_augmentations: Augmentations per puzzle (Mini-ARC, Concept-ARC, RE-ARC)
        arc_augmentations: Augmentations per puzzle (ARC-AGI-1, ARC-AGI-2)
        shard_size: Samples per shard file
        max_seq_length: Filter out puzzles > this estimated token count
        output_name: Output directory name
    """
    data_dir = get_data_dir()
    
    print(f"=== Generating Phase 2 Training Data ===")
    print(f"  RE-ARC base puzzles: {num_rearc_base_puzzles}")
    print(f"  Augmentations (Mini/Concept/RE-ARC): {num_augmentations}")
    print(f"  Augmentations (ARC-AGI-1/2): {arc_augmentations}")
    print(f"  Max sequence length: {max_seq_length:,}")
    print(f"  Shard size: {shard_size}")
    print()
    
    # Generate RE-ARC data (need fresh generation for the puzzles)
    print("Step 1: Generating RE-ARC examples...")
    generate_rearc_data(rearc_cnt=200, delete=True)  # Generate enough examples
    
    # Setup output directory
    output_path = data_dir / output_name
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    
    # Collect all puzzles
    print("\nStep 2: Loading and augmenting puzzles...")
    all_puzzles = []
    skipped_count = 0
    
    # Mini-ARC
    print("  Loading Mini-ARC...")
    mini_arc_count = 0
    for puzzle, puzzle_id in iter_mini_arc_puzzles(num_augs=num_augmentations):
        est_tokens = estimate_puzzle_tokens(puzzle['puzzle'])
        if est_tokens <= max_seq_length:
            all_puzzles.append((puzzle['puzzle'], puzzle_id))
            mini_arc_count += 1
        else:
            skipped_count += 1
    print(f"    Added {mini_arc_count} Mini-ARC samples")
    
    # Concept-ARC
    print("  Loading Concept-ARC...")
    concept_arc_count = 0
    for puzzle, puzzle_id in iter_concept_arc_puzzles(num_augs=num_augmentations):
        est_tokens = estimate_puzzle_tokens(puzzle['puzzle'])
        if est_tokens <= max_seq_length:
            all_puzzles.append((puzzle['puzzle'], puzzle_id))
            concept_arc_count += 1
        else:
            skipped_count += 1
    print(f"    Added {concept_arc_count} Concept-ARC samples")
    
    # RE-ARC (limited)
    print("  Loading RE-ARC...")
    rearc_count = 0
    for puzzle, puzzle_id in iter_rearc_puzzles_limited(
        num_base_puzzles=num_rearc_base_puzzles,
        num_augs=num_augmentations
    ):
        est_tokens = estimate_puzzle_tokens(puzzle['puzzle'])
        if est_tokens <= max_seq_length:
            all_puzzles.append((puzzle['puzzle'], puzzle_id))
            rearc_count += 1
        else:
            skipped_count += 1
    print(f"    Added {rearc_count} RE-ARC samples")
    
    # ARC-AGI-1
    print("  Loading ARC-AGI-1...")
    arc1_count = 0
    for puzzle, puzzle_id in iter_arc_puzzles(variant=1, num_augs=arc_augmentations):
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
    for puzzle, puzzle_id in iter_arc_puzzles(variant=2, num_augs=arc_augmentations):
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
    print("\nStep 3: Shuffling and writing shards...")
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
    print(f"  - Mini-ARC: {mini_arc_count:,}")
    print(f"  - Concept-ARC: {concept_arc_count:,}")
    print(f"  - RE-ARC: {rearc_count:,}")
    print(f"  - ARC-AGI-1: {arc1_count:,}")
    print(f"  - ARC-AGI-2: {arc2_count:,}")
    
    return output_path


def main():
    """Generate Phase 2 data with default settings."""
    return generate_phase2_data(
        num_rearc_base_puzzles=100,
        num_augmentations=10,
        arc_augmentations=15,
        shard_size=1000,
        max_seq_length=1024 * 8,
    )

