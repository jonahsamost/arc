"""
ARC Dataset and DataLoader utilities.

Supports:
- Iterable dataset for streaming from shards
- Proper collation for 2D tokenizer outputs (variable length sequences)
- Multi-worker data loading with shard distribution
"""

import torch
import json
import glob
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from typing import Dict, List, Optional, Any
from src.data.tokenizer_2d import Arc2DTokenizer, IGNORE_INDEX
from src.data.tokenizer_1d import ArcBaselineTokenizer


class ArcDataset(IterableDataset):
    """
    Iterable dataset for ARC puzzles stored in JSONL shards.
    
    Each shard contains lines of: [task_data_dict, filename]
    where task_data_dict has 'puzzle' key with train/test pairs.
    """
    
    def __init__(
        self,
        data_dir: str,
        tokenizer: Optional[Any] = None,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        baseline_1d: bool = False,
        raw: bool = False,
        shuffle_shards: bool = True,
        fim_ratio: float = 0.0,  # Ratio of samples to convert to FIM format
        max_seq_length: int = 16384,  # Skip samples longer than this
    ):
        """
        Args:
            data_dir: Directory containing .jsonl shard files
            tokenizer: Pre-initialized tokenizer (preferred for 2D)
            model_name: Model name for tokenizer initialization
            baseline_1d: If True, use 1D baseline tokenizer
            raw: If True, yield raw data without tokenization
            shuffle_shards: If True, shuffle shard order per epoch
            fim_ratio: Fraction of samples to format as FIM (0.0 = all NTP, 1.0 = all FIM)
            max_seq_length: Skip samples longer than this (OOM protection)
        """
        self.data_dir = data_dir
        self.shard_files = sorted(glob.glob(f"{data_dir}/*.jsonl"))
        self.raw = raw
        self.shuffle_shards = shuffle_shards
        self.fim_ratio = fim_ratio
        self.max_seq_length = max_seq_length
        
        if not self.shard_files:
            raise FileNotFoundError(f"No .jsonl files found in {data_dir}")
        
        # Initialize or use provided tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif baseline_1d:
            self.tokenizer = ArcBaselineTokenizer(model_name=model_name)
        else:
            self.tokenizer = Arc2DTokenizer(model_name=model_name)
        
        print(f"ArcDataset initialized with {len(self.shard_files)} shards from {data_dir}")
        if fim_ratio > 0:
            print(f"  FIM ratio: {fim_ratio:.0%} of samples will use FIM format")
    
    def __iter__(self):
        worker_info = get_worker_info()
        
        if worker_info is None:
            # Single-process loading
            my_shards = self.shard_files.copy()
        else:
            # Multi-process: distribute shards across workers
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            my_shards = [
                f for i, f in enumerate(self.shard_files)
                if i % total_workers == worker_id
            ]
        
        # Optional shuffle
        if self.shuffle_shards:
            import random
            random.shuffle(my_shards)
        
        for file_path in my_shards:
            with open(file_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        data, filename = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    
                    if self.raw:
                        yield data, filename
                    else:
                        try:
                            # Randomly choose between NTP and FIM based on fim_ratio
                            import random
                            use_fim = self.fim_ratio > 0 and random.random() < self.fim_ratio
                            
                            if use_fim and hasattr(self.tokenizer, 'build_fim_sample'):
                                sample = self.tokenizer.build_fim_sample(data['puzzle'])
                            else:
                                sample = self.tokenizer.build_sample(data['puzzle'])
                            
                            # Note: Long sequences are now handled by chunked cross-entropy
                            # Only skip extremely long sequences (>16K) as a safety limit
                            seq_len = sample["input_ids"].shape[0]
                            if seq_len > self.max_seq_length:
                                print(f"[Warning] Skipping {filename}: {seq_len} tokens > max {self.max_seq_length}")
                                continue
                            
                            yield sample, filename
                        except Exception as e:
                            # Skip malformed puzzles
                            print(f"Warning: Failed to tokenize {filename}: {e}")
                            continue


def collate_arc_2d(batch: List[tuple]) -> Dict[str, torch.Tensor]:
    """
    Collate function for Arc2DTokenizer outputs.
    
    Handles variable-length sequences by padding to max length in batch.
    
    Input batch format: List of (sample_dict, filename) tuples
    where sample_dict has: input_ids, labels, attention_mask, pos_1d, pos_2d, grid_mode, output_mask
    
    Output: Dict with batched tensors, all padded to max_seq_len
    """
    samples = [item[0] for item in batch]
    filenames = [item[1] for item in batch]
    
    # Find max sequence length in batch
    max_len = max(s["input_ids"].size(0) for s in samples)
    batch_size = len(samples)
    
    # Prepare output tensors
    input_ids = torch.full((batch_size, max_len), 0, dtype=torch.long)  # Pad with 0
    labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    pos_1d = torch.zeros((batch_size, max_len), dtype=torch.long)
    pos_2d = torch.zeros((batch_size, max_len, 2), dtype=torch.long)
    grid_mode = torch.zeros((batch_size, max_len), dtype=torch.long)
    output_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    
    for i, sample in enumerate(samples):
        seq_len = sample["input_ids"].size(0)
        
        input_ids[i, :seq_len] = sample["input_ids"]
        labels[i, :seq_len] = sample["labels"]
        attention_mask[i, :seq_len] = sample["attention_mask"]
        pos_1d[i, :seq_len] = sample["pos_1d"]
        pos_2d[i, :seq_len] = sample["pos_2d"]
        grid_mode[i, :seq_len] = sample["grid_mode"]
        output_mask[i, :seq_len] = sample["output_mask"]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "pos_1d": pos_1d,
        "pos_2d": pos_2d,
        "grid_mode": grid_mode,
        "output_mask": output_mask,
        "filenames": filenames,
    }


def collate_arc_1d(batch: List[tuple]) -> Dict[str, torch.Tensor]:
    """
    Collate function for ArcBaselineTokenizer (1D) outputs.
    """
    samples = [item[0] for item in batch]
    filenames = [item[1] for item in batch]
    
    max_len = max(s["input_ids"].size(0) for s in samples)
    batch_size = len(samples)
    
    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    
    for i, sample in enumerate(samples):
        seq_len = sample["input_ids"].size(0)
        input_ids[i, :seq_len] = sample["input_ids"]
        labels[i, :seq_len] = sample["labels"]
        attention_mask[i, :seq_len] = sample["attention_mask"]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "filenames": filenames,
    }


def create_dataloader(
    data_dir: str,
    tokenizer: Optional[Any] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    baseline_1d: bool = False,
    shuffle_shards: bool = True,
    fim_ratio: float = 0.0,
    max_seq_length: int = 4096,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for ARC training.
    
    Args:
        data_dir: Directory containing .jsonl shards
        tokenizer: Pre-initialized tokenizer
        batch_size: Batch size
        num_workers: Number of data loading workers
        baseline_1d: Use 1D tokenizer
        shuffle_shards: Shuffle shard order
        fim_ratio: Fraction of samples to use FIM format (0.0-1.0)
        max_seq_length: Skip samples longer than this (OOM protection)
        **kwargs: Additional DataLoader kwargs
    
    Returns:
        Configured DataLoader
    """
    dataset = ArcDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        baseline_1d=baseline_1d,
        shuffle_shards=shuffle_shards,
        fim_ratio=fim_ratio,
        max_seq_length=max_seq_length,
    )
    
    collate_fn = collate_arc_1d if baseline_1d else collate_arc_2d
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        **kwargs,
    )
