import random
import json
import shutil
from pathlib import Path
from typing import Iterator, Tuple, Any, Optional
from tqdm import tqdm

from src.data.kant import generate_kant_dataset
from src.data.build_arc_dataset import (
    generate_rearc_data, get_data_dir,
    load_kant_arc, get_rearc_path
)
from src.data.common import ARCAugmenter


def estimate_puzzle_tokens(puzzle: dict) -> int:
    """
    Quick estimate of token count for a puzzle without full tokenization.
    
    This is ~10x faster than actual tokenization but slightly overestimates.
    Use for fast filtering during data generation.
    """
    total_pixels = 0
    
    # Count train examples
    for example in puzzle.get('train', []):
        if 'input' in example:
            inp = example['input']
            total_pixels += len(inp) * len(inp[0]) if inp else 0
        if 'output' in example:
            out = example['output']
            total_pixels += len(out) * len(out[0]) if out else 0
    
    # Count test examples
    for example in puzzle.get('test', []):
        if 'input' in example:
            inp = example['input']
            total_pixels += len(inp) * len(inp[0]) if inp else 0
        if 'output' in example:
            out = example['output']
            total_pixels += len(out) * len(out[0]) if out else 0
    
    # Add overhead for metadata, separators, etc. (~20% overhead)
    return int(total_pixels * 1.2)


def iter_rearc_puzzles(
    num_augs: int = 10,
    chunk_size: int = 5000,
    max_puzzles: int = None,
) -> Iterator[Tuple[Any, str]]:
    """
    Generator that yields RE-ARC puzzles in chunks to avoid memory blowup.
    
    Yields:
        (puzzle_dict, puzzle_id) tuples
    """
    from src.data.build_arc_dataset import sample_puzzle_size
    
    re_arc_path = get_rearc_path()
    arc_aug = ARCAugmenter()
    path = re_arc_path / 'tasks'
    
    total_yielded = 0
    
    for filepath in path.rglob('*.json'):
        if max_puzzles and total_yielded >= max_puzzles:
            return
            
        if filepath.is_file():
            with open(filepath, 'r') as fd:
                all_examples = json.loads(fd.read())
            
            if len(all_examples) < 3:
                continue
            
            random.shuffle(all_examples)
            
            # Create multiple puzzles from the pool of examples
            idx = 0
            puzzle_num = 0
            
            while idx < len(all_examples):
                if max_puzzles and total_yielded >= max_puzzles:
                    return
                
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
                puzzle_id = f"{filepath.stem}_p{puzzle_num}"
                
                # Apply augmentations and yield each
                for aug_puzzle in arc_aug.augment_puzzle(task, num_augmentations=num_augs):
                    if max_puzzles and total_yielded >= max_puzzles:
                        return
                    yield (aug_puzzle, f"{puzzle_id}_aug{total_yielded % num_augs}")
                    total_yielded += 1
                
                idx += num_train + num_test
                puzzle_num += 1


def iter_kant_puzzles(
    kant_path: Path,
    max_puzzles: int = None,
) -> Iterator[Tuple[Any, str]]:
    """
    Generator that yields KANT puzzles.
    
    Yields:
        (puzzle_dict, puzzle_id) tuples
    """
    total_yielded = 0
    
    for filepath in kant_path.rglob('*.json'):
        if max_puzzles and total_yielded >= max_puzzles:
            return
            
        if filepath.is_file():
            with open(filepath, 'r') as fd:
                data = json.loads(fd.read())
            
            yield ({'puzzle': data}, str(filepath.stem))
            total_yielded += 1


def generate_phase1_streaming(
    rearc_examples_per_task: int = 3000,
    num_augmentations: int = 10,
    include_kant: bool = True,
    kant_samples: int = 400_000,
    rearc_samples: int = 600_000,
    chunk_size: int = 5000,
    shard_size: int = 1000,
    max_seq_length: Optional[int] = None,  # Filter out puzzles > this token count
):
    """
    Generate training data using streaming to avoid memory blowup.
    
    Loads puzzles in chunks and writes mixed shards containing both
    RE-ARC and KANT puzzles.
    
    Args:
        max_seq_length: If set, skip puzzles estimated to exceed this token count.
                       Uses fast estimation (~10x faster than full tokenization).
    """
    data_dir = get_data_dir()
    
    print(f"=== Generating Phase 1 Training Data (Streaming) ===")
    print(f"  RE-ARC samples: {rearc_samples:,}")
    print(f"  KANT samples: {kant_samples:,}")
    print(f"  Chunk size: {chunk_size:,}")
    print(f"  data dir: {data_dir}")
    if max_seq_length:
        print(f"  Max sequence length: {max_seq_length:,} (filtering enabled)")
    print()
    
    # Generate RE-ARC data
    print("Step 1: Generating RE-ARC examples...")
    generate_rearc_data(rearc_cnt=rearc_examples_per_task, delete=True)
    
    # Generate KANT data
    kant_path = None
    if include_kant:
        print("Step 2: Generating KANT puzzles...")
        kant_path = data_dir / 'arc_data_training/kant'
        generate_kant_dataset(n=kant_samples, output=kant_path)
    
    # Setup output directory
    output_path = data_dir / 'train_data_phase1'
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    
    # Create iterators
    print("\nStep 3: Streaming and sharding puzzles...")
    rearc_iter = iter_rearc_puzzles(
        num_augs=num_augmentations,
        max_puzzles=rearc_samples,
    )
    
    kant_iter = iter_kant_puzzles(kant_path, max_puzzles=kant_samples) if include_kant else iter([])
    
    # Calculate ratio for mixing (e.g., 60% RE-ARC, 40% KANT)
    total_samples = rearc_samples + (kant_samples if include_kant else 0)
    rearc_ratio = rearc_samples / total_samples
    
    # Stream and write shards
    current_shard = 0
    current_buffer = []
    total_written = 0
    rearc_exhausted = False
    kant_exhausted = False
    
    pbar = tqdm(total=total_samples, desc="Writing shards")
    
    skipped_count = 0
    
    while not (rearc_exhausted and kant_exhausted):
        # Fill buffer with mixed samples
        while len(current_buffer) < chunk_size and not (rearc_exhausted and kant_exhausted):
            # Decide which source to pull from based on ratio
            use_rearc = random.random() < rearc_ratio
            
            puzzle = None
            if use_rearc and not rearc_exhausted:
                try:
                    puzzle = next(rearc_iter)
                except StopIteration:
                    rearc_exhausted = True
            elif not kant_exhausted:
                try:
                    puzzle = next(kant_iter)
                except StopIteration:
                    kant_exhausted = True
            elif not rearc_exhausted:
                # KANT exhausted, pull from RE-ARC
                try:
                    puzzle = next(rearc_iter)
                except StopIteration:
                    rearc_exhausted = True
            
            # Apply max_seq_length filter if enabled
            if puzzle is not None:
                puzzle_data, puzzle_id = puzzle
                
                # Handle both {'puzzle': ...} and direct puzzle format
                actual_puzzle = puzzle_data.get('puzzle', puzzle_data)
                
                if max_seq_length:
                    est_tokens = estimate_puzzle_tokens(actual_puzzle)
                    if est_tokens > max_seq_length:
                        skipped_count += 1
                        continue  # Skip this puzzle
                
                current_buffer.append(puzzle)
        
        # Shuffle buffer for good mixing within shard
        random.shuffle(current_buffer)
        
        # Write shards from buffer
        while len(current_buffer) >= shard_size:
            shard_data = current_buffer[:shard_size]
            current_buffer = current_buffer[shard_size:]
            
            shard_path = output_path / f"shard_{current_shard}.jsonl"
            with open(shard_path, 'w') as f:
                for puzzle, puzzle_id in shard_data:
                    f.write(json.dumps([puzzle, puzzle_id]) + '\n')
            
            total_written += len(shard_data)
            pbar.update(len(shard_data))
            current_shard += 1
    
    # Write remaining buffer as final shard
    if current_buffer:
        shard_path = output_path / f"shard_{current_shard}.jsonl"
        with open(shard_path, 'w') as f:
            for puzzle, puzzle_id in current_buffer:
                f.write(json.dumps([puzzle, puzzle_id]) + '\n')
        total_written += len(current_buffer)
        pbar.update(len(current_buffer))
    
    pbar.close()
    
    print(f"\n=== Done! ===")
    print(f"Output: {output_path}")
    print(f"Total samples: {total_written:,}")
    print(f"Total shards: {current_shard + 1}")
    if max_seq_length:
        print(f"Skipped (too long): {skipped_count:,}")
    
    return output_path


def generate_1m_samples(max_seq_length: int = 1024 * 10):
    """
    Quick function to generate ~1M samples.
    
    Args:
        max_seq_length: Filter out puzzles estimated to exceed this token count.
                       Default 12288 matches training config.
    """
    return generate_phase1_streaming(
        rearc_examples_per_task=2500,
        num_augmentations=10,
        include_kant=True,
        kant_samples=400_000,
        rearc_samples=600_000,
        chunk_size=10000,  # Keep 10K in memory at a time
        shard_size=1000,
        max_seq_length=max_seq_length,
    )


