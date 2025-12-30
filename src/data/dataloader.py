import torch
import json
import glob
import math
from torch.utils.data import IterableDataset, get_worker_info
from src.data.tokenizer_2d import Arc2DTokenizer
from src.data.tokenizer_1d import ArcBaselineTokenizer


class ArcDataset(IterableDataset):
    def __init__(
        self, data_dir: str, model_name: str = 'Qwen/Qwen2.5-Coder-7B-Instruct',
        baseline_1d: bool = False, raw: bool = False
    ):
        self.data_dir = data_dir
        self.shard_files = sorted(glob.glob(f"{data_dir}/*.jsonl"))
        self.raw = raw
        if baseline_1d:
            self.tokenizer = ArcBaselineTokenizer(model_name=model_name)
        else:
            self.tokenizer = Arc2DTokenizer()
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
                    datas = json.loads(line)
                    data, filename = datas
                    if self.raw:
                        yield data, filename
                    else:
                        yield self.tokenizer.build_sample(data['puzzle']), filename
