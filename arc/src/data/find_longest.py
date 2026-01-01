"""
Find the longest tokenized sequence in the dataset.

Usage:
    python -m src.data.find_longest /root/arc_data/train_data/
"""

import json
import glob
import sys
from pathlib import Path
from tqdm import tqdm

from src.data.tokenizer_2d import Arc2DTokenizer


def estimate_puzzle_size(puzzle: dict) -> int:
    """
    Quick estimate of puzzle size without tokenization.
    Counts total grid cells across all pairs.
    """
    total_cells = 0
    
    for pair in puzzle.get('train', []):
        if 'input' in pair:
            total_cells += len(pair['input']) * len(pair['input'][0]) if pair['input'] else 0
        if 'output' in pair:
            total_cells += len(pair['output']) * len(pair['output'][0]) if pair['output'] else 0
    
    for pair in puzzle.get('test', []):
        if 'input' in pair:
            total_cells += len(pair['input']) * len(pair['input'][0]) if pair['input'] else 0
        if 'output' in pair:
            total_cells += len(pair['output']) * len(pair['output'][0]) if pair['output'] else 0
    
    return total_cells


def get_puzzle_dims(puzzle: dict) -> str:
    """Get human-readable dimensions of puzzle grids."""
    dims = []
    for i, pair in enumerate(puzzle.get('train', [])):
        in_shape = f"{len(pair['input'])}x{len(pair['input'][0])}" if pair.get('input') else "?"
        out_shape = f"{len(pair['output'])}x{len(pair['output'][0])}" if pair.get('output') else "?"
        dims.append(f"train[{i}]: {in_shape} -> {out_shape}")
    
    for i, pair in enumerate(puzzle.get('test', [])):
        in_shape = f"{len(pair['input'])}x{len(pair['input'][0])}" if pair.get('input') else "?"
        out_shape = f"{len(pair['output'])}x{len(pair['output'][0])}" if pair.get('output') else "?"
        dims.append(f"test[{i}]: {in_shape} -> {out_shape}")
    
    return ", ".join(dims)


def find_longest_sequences(data_dir: str, top_k: int = 10):
    """
    Find the longest tokenized sequences in the dataset.
    
    Args:
        data_dir: Directory containing .jsonl shards
        top_k: Number of longest sequences to report
    """
    shard_files = sorted(glob.glob(f"{data_dir}/*.json*"))
    
    if not shard_files:
        print(f"No .jsonl files found in {data_dir}")
        return
    
    print(f"Found {len(shard_files)} shards in {data_dir}")
    print(f"Scanning for longest sequences...\n")
    
    # First pass: estimate sizes without tokenization
    candidates = []  # (estimated_size, puzzle_data, filename)
    total_puzzles = 0
    max_estimate = 0

    # for shard_file in tqdm(shard_files, desc="scanning shards"):
    #     with open(shard_file, 'r') as f:
    #         try:
    #             puzzle = json.loads(f.read())
    #             est_size = estimate_puzzle_size(puzzle)
    #             max_estimate = max(max_estimate, est_size)
    #             continue
    #             candidates.append((est_size, puzzle, filename))
    #             total_puzzles += 1
    #         except Exception as e:
    #             continue

    above = 0
    max_seq_length = 12288
    for shard_file in tqdm(shard_files, desc="scanning shards"):
        with open(shard_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data, filename = json.loads(line)
                    puzzle = data['puzzle']
                    est_size = estimate_puzzle_size(puzzle)
                    if est_size >= max_seq_length:
                        above += 1
                    max_estimate = max(max_estimate, est_size)
                    continue
                    candidates.append((est_size, puzzle, filename))
                    total_puzzles += 1
                except (json.jsondecodeerror, valueerror, keyerror):
                    continue
    
    print(f"\nTotal puzzles: {total_puzzles}")
    
    # Sort by estimated size, take top candidates for tokenization
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidates[:top_k * 5]  # Check more than top_k to be safe
    
    print(f"Tokenizing top {len(top_candidates)} candidates by estimated size...\n")
    
    # Initialize tokenizer
    tokenizer = Arc2DTokenizer()
    
    # Second pass: tokenize top candidates
    results = []  # (token_count, estimated_size, filename, dims)
    
    for est_size, puzzle, filename in tqdm(top_candidates, desc="Tokenizing"):
        try:
            sample = tokenizer.build_sample(puzzle)
            token_count = sample['input_ids'].shape[0]
            dims = get_puzzle_dims(puzzle)
            results.append((token_count, est_size, filename, dims))
        except Exception as e:
            print(f"  Warning: Failed to tokenize {filename}: {e}")
    
    # Sort by actual token count
    results.sort(key=lambda x: x[0], reverse=True)
    
    # Report top K
    print("\n" + "=" * 80)
    print(f"TOP {top_k} LONGEST TOKENIZED SEQUENCES")
    print("=" * 80)
    
    for i, (token_count, est_size, filename, dims) in enumerate(results[:top_k], 1):
        print(f"\n#{i}: {filename}")
        print(f"    Tokens: {token_count:,}")
        print(f"    Est. cells: {est_size:,}")
        print(f"    Grids: {dims}")
    
    # Stats
    if results:
        token_counts = [r[0] for r in results]
        print("\n" + "=" * 80)
        print("STATISTICS (from top candidates)")
        print("=" * 80)
        print(f"  Max tokens: {max(token_counts):,}")
        print(f"  Min tokens: {min(token_counts):,}")
        print(f"  Avg tokens: {sum(token_counts) / len(token_counts):,.0f}")
    
    return results


def main():
    data_dir = "/root/arc_data/train_data/"
    top_k = 5
    find_longest_sequences(data_dir, top_k=top_k)



