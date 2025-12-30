import torch
import json
import glob
import math
from torch.utils.data import IterableDataset, get_worker_info

class ArcDataset(IterableDataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.shard_files = sorted(glob.glob(f"{data_dir}/*.jsonl"))
        if not self.shard_files:
            raise FileNotFoundError(f"No .jsonl files found in {data_dir}")

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            my_shards = self.shard_files
        else:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            
            my_shards = [
                f for i, f in enumerate(self.shard_files) 
                if i % total_workers == worker_id
            ]

        for file_path in my_shards:
            with open(file_path, 'r') as f:
                for line in f:
                    if not line.strip(): 
                        continue
                    
                    data = json.loads(line)
                    yield self.transform(data)

    def transform(self, raw_dict):
        # Your padding / tensor conversion logic
        # Return dict of tensors
        return raw_dict