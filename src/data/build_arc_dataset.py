import shutil
import json
import random
from pathlib import Path
from src.data.common import ARCAugmenter
from src.data.re_arc.main import generate_dataset


def generate_data(
    training: bool = True, output_path: str = 'train_data', num_augs: int = 10, rearc_cnt: int = 10
):
    BASE_DIR = Path(__file__).resolve().parent
    re_arc_dpath = Path(BASE_DIR / 're_arc_data')
    if re_arc_dpath.exists():
        try:
            shutil.rmtree(re_arc_dpath)
            print(f"Cleaned up existing directory: {re_arc_dpath}")
        except OSError as e:
            print(f"Error deleting {re_arc_dpath}: {e}")
    generate_dataset(path=re_arc_dpath, n_examples=rearc_cnt)
    loaded_puzzles = load_puzzles(training=training, re_arc_path=re_arc_dpath)
    arc_aug = ARCAugmenter()
    aug_puzzles = [
        x for puzz in loaded_puzzles
        for x in arc_aug.augment_puzzle(puzz, num_augmentations=num_augs)
    ]

    current_shard = 0
    current_count = 0
    output_path = Path(BASE_DIR / output_path)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    file_handle = open(output_path / f"shard_{current_shard}.jsonl", 'w')
    for aug in aug_puzzles:
        file_handle.write(json.dumps(aug) + '\n')
        current_count += 1
        
        if current_count >= 1000:
            file_handle.close()
            current_shard += 1
            file_handle = open(output_path / f"shard_{current_shard}.jsonl", 'w')
            current_count = 0 


def load_puzzles(training: bool = True, re_arc_path: str = None):
    puzzles = []
    BASE_DIR = Path(__file__).resolve().parent
    dpath = Path(BASE_DIR / 'arc_data_training' if training else 'arc_data_eval')
    for filepath in dpath.rglob('*.json'):
        if filepath.is_file():
            with open(filepath, 'r') as fd:
                data = json.loads(fd.read())
                puzzles.append(data)
    
    if re_arc_path:
        path = re_arc_path / 'tasks'
        for filepath in path.rglob('*.json'):
            if filepath.is_file():
                with open(filepath, 'r') as fd:
                    examples = json.loads(fd.read())
                    random.shuffle(examples)
                    leftover = examples
                    while leftover:
                        num_train = random.randint(2, 5)
                        num_test = 1
                        cnt = num_train + num_test
                        if cnt >= len(leftover):
                            break
                        selection = leftover[:cnt]
                        task = {
                            "train": selection[:cnt - 1],
                            "test": selection[cnt - 1:]
                        }
                        leftover = leftover[cnt:]
                        puzzles.append(task)
    return puzzles