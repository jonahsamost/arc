import shutil
import json
import random
from pathlib import Path

from torch.utils import data
from src.data.common import ARCAugmenter
from src.data.re_arc.main import generate_dataset


def baseline_eval_all_arc_1d(variant1=True, variant2=True):
    print('Getting baseline data')
    puzzles = get_arc_puzzles(variant1=variant1, variant2=variant2)
    print(f'Loaded {len(puzzles)} puzzles')
    aug_puzzles = [({'puzzle': x}, filepath) for x, filepath in puzzles]
    output_path = shard_puzzles(aug_puzzles, training=False)
    return output_path


def baseline_eval_arc1_1d():
    return baseline_eval_all_arc_1d(variant1=True, variant2=False)


def baseline_eval_arc2_1d():
    return baseline_eval_all_arc_1d(variant1=False, variant2=True)


def baseline_eval_all_arc_2d(variant1=True, variant2=True):
    print('Getting baseline data')
    puzzles = get_arc_puzzles(variant1=variant1, variant2=variant2)
    arc_aug = ARCAugmenter()
    aug_puzzles = [
        (x, filepath) for puzz, filepath in puzzles
        for x in arc_aug.augment_puzzle(puzz, augment=False, jitter=False)
    ]
    output_path = shard_puzzles(aug_puzzles, training=False)
    return output_path


def baseline_eval_arc1_2d():
    print('Getting baseline data')
    return baseline_eval_all_arc_2d(variant1=True, variant2=False)


def baseline_eval_arc2_2d():
    print('Getting baseline data')
    return baseline_eval_all_arc_2d(variant1=False, variant2=True)


def get_arc_puzzles(variant1=True, variant2=True):
    puzzles = []
    if variant1:
        puzzles += load_arc_puzzles(variant=1, training=False)
    if variant2:
        puzzles += load_arc_puzzles(variant=2, training=False)
    return puzzles


def get_rearc_path():
    data_dir = get_data_dir()
    re_arc_dpath = data_dir / 'arc_data_training/re_arc_data'
    return re_arc_dpath


def generate_rearc_data(rearc_cnt: int = 10, delete=True):
    re_arc_dpath = get_rearc_path()
    if delete and re_arc_dpath.exists():
        try:
            shutil.rmtree(re_arc_dpath)
            print(f"Cleaned up existing directory: {re_arc_dpath}")
        except OSError as e:
            print(f"Error deleting {re_arc_dpath}: {e}")
    generate_dataset(path=re_arc_dpath, n_examples=rearc_cnt)
    return re_arc_dpath


def load_rearc_data(num_augs: int = 10):
    re_arc_dpath = get_rearc_path()
    arc_aug = ARCAugmenter()
    puzzles = load_rearc_puzzles(re_arc_dpath)
    aug_puzzles = [
        (x, filepath) for puzz, filepath in puzzles
        for x in arc_aug.augment_puzzle(puzz, num_augmentations=num_augs)
    ]
    return aug_puzzles



def shard_puzzles(puzzles, training: bool=True, output_name: str = ''):
    data_dir = get_data_dir()
    current_shard = 0
    current_count = 0
    if output_name:
        out_path = output_name
    else:
        out_path = 'train_data' if training else 'eval_data'

    output_path = Path(data_dir / out_path)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    file_handle = open(output_path / f"shard_{current_shard}.jsonl", 'w')
    for aug in puzzles:
        file_handle.write(json.dumps(aug) + '\n')
        current_count += 1
        
        if current_count >= 1000:
            file_handle.close()
            current_shard += 1
            file_handle = open(output_path / f"shard_{current_shard}.jsonl", 'w')
            current_count = 0 
    return output_path


def sample_puzzle_size(compact: bool = True) -> tuple[int, int]:
    """
    Sample number of training and test examples with a realistic distribution.
    
    Args:
        compact: If True, use smaller puzzle sizes to reduce token count.
                 Matches real ARC distribution better (most have 3 train, 1 test).
    
    Returns:
        (num_train, num_test)
    """
    if compact:
        # Compact distribution - matches real ARC puzzles better
        # Real ARC: ~70% have 3 train examples, ~25% have 2-4, rest have more
        train_weights = {
            2: 15,   # 15%
            3: 50,   # 50% - most common in real ARC
            4: 25,   # 25%
            5: 10,   # 10%
        }
        # Real ARC: ~95% have 1 test example
        test_weights = {
            1: 95,   # 95%
            2: 5,    # 5%
        }
    else:
        # Original distribution - more varied sizes
        train_weights = {
            2: 10,   # 5%
            3: 25,   # 25%
            4: 30,   # 30%
            5: 20,   # 20%
            6: 10,   # 10%
            7: 5,    # 5%
        }
        test_weights = {
            1: 80,   # 80%
            2: 15,   # 15%
            3: 5,    # 5%
        }
    
    train_choices = list(train_weights.keys())
    train_probs = [train_weights[k] for k in train_choices]
    num_train = random.choices(train_choices, weights=train_probs, k=1)[0]
    
    test_choices = list(test_weights.keys())
    test_probs = [test_weights[k] for k in test_choices]
    num_test = random.choices(test_choices, weights=test_probs, k=1)[0]
    
    return num_train, num_test


def load_rearc_puzzles(re_arc_path, examples_per_puzzle: int = None):
    """
    Load RE-ARC generated examples and create puzzles with varied sizes.
    
    Args:
        re_arc_path: Path to RE-ARC data
        examples_per_puzzle: If set, use fixed size. If None, sample from distribution.
    
    Returns:
        List of (puzzle_dict, filepath) tuples
    """
    puzzles = []
    path = re_arc_path / 'tasks'
    
    for filepath in path.rglob('*.json'):
        if filepath.is_file():
            with open(filepath, 'r') as fd:
                all_examples = json.loads(fd.read())
            
            if len(all_examples) < 3:
                # Need at least 2 train + 1 test
                continue
            
            random.shuffle(all_examples)
            
            # Create multiple puzzles from the pool of examples
            idx = 0
            puzzle_num = 0
            
            while idx < len(all_examples):
                # Sample puzzle size
                if examples_per_puzzle:
                    num_train = examples_per_puzzle - 1
                    num_test = 1
                else:
                    num_train, num_test = sample_puzzle_size()
                
                total_needed = num_train + num_test
                
                # Check if we have enough examples left
                if idx + total_needed > len(all_examples):
                    # Use remaining examples if enough for minimal puzzle
                    remaining = len(all_examples) - idx
                    if remaining >= 3:  # At least 2 train + 1 test
                        num_test = min(num_test, remaining // 3)
                        num_train = remaining - num_test
                    else:
                        break
                
                # Extract examples for this puzzle
                train_examples = all_examples[idx:idx + num_train]
                test_examples = all_examples[idx + num_train:idx + num_train + num_test]
                
                task = {
                    "train": train_examples,
                    "test": test_examples
                }
                
                puzzle_id = f"{filepath.stem}_p{puzzle_num}"
                puzzles.append((task, puzzle_id))
                
                idx += num_train + num_test
                puzzle_num += 1
    
    return puzzles


def load_concept_arc():
    data_dir = get_data_dir()
    dpath = Path(data_dir / 'arc_data_training/concept_arc')
    return load_puzzles(dpath)

def load_mini_arc():
    data_dir = get_data_dir()
    dpath = Path(data_dir / 'arc_data_training/mini_arc')
    return load_puzzles(dpath)


def load_kant_arc():
    data_dir = get_data_dir()
    dpath = Path(data_dir / 'arc_data_training/kant')
    return load_puzzles(dpath)


def load_arc_puzzles(variant: int = 1, training: bool = True):
    data_dir = get_data_dir()
    dpath = Path(data_dir / ('arc_data_training' if training else 'arc_data_eval'))
    dpath = dpath / f'arc_agi_{variant}'
    arc1 = variant == 1
    arc2 = variant == 2
    print(f'Using arc1: {arc1}, arc2: {arc2}, dpath: {dpath}')
    return load_puzzles(dpath)


def load_puzzles(dpath):
    puzzles = []
    for filepath in dpath.rglob('*.json'):
        if filepath.is_file():
            with open(filepath, 'r') as fd:
                data = json.loads(fd.read())
                puzzles.append((data, str(filepath)))
    return puzzles


def get_data_dir():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    return BASE_DIR / 'arc_data'
