import shutil
import json
import random
from pathlib import Path
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


def generate_rearc_data(rearc_cnt: int = 10, num_augs: int = 10):
    BASE_DIR = Path(__file__).resolve().parent
    re_arc_dpath = Path(BASE_DIR / 're_arc_data')
    if re_arc_dpath.exists():
        try:
            shutil.rmtree(re_arc_dpath)
            print(f"Cleaned up existing directory: {re_arc_dpath}")
        except OSError as e:
            print(f"Error deleting {re_arc_dpath}: {e}")
    generate_dataset(path=re_arc_dpath, n_examples=rearc_cnt)

    arc_aug = ARCAugmenter()
    puzzles = load_rearc_puzzles(re_arc_dpath)
    aug_puzzles = [
        (x, filepath) for puzz, filepath in puzzles
        for x in arc_aug.augment_puzzle(puzz, num_augmentations=num_augs)
    ]
    output_path = shard_puzzles(aug_puzzles, training=True, output_name='rearc_train_data')
    return output_path



def shard_puzzles(puzzles, training: bool=True, output_name: str = ''):
    BASE_DIR = Path(__file__).resolve().parent
    current_shard = 0
    current_count = 0
    if output_name:
        out_path = output_name
    else:
        out_path = 'train_data' if training else 'eval_data'

    output_path = Path(BASE_DIR / out_path)
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


def load_rearc_puzzles(re_arc_path):
    puzzles = []
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
                    puzzles.append((task, str(filepath)))
    return puzzles


def load_concept_arc():
    BASE_DIR = Path(__file__).resolve().parent
    dpath = Path(BASE_DIR / 'arc_data_training/concept_arc')
    return load_puzzles(dpath)

def load_mini_arc():
    BASE_DIR = Path(__file__).resolve().parent
    dpath = Path(BASE_DIR / 'arc_data_training/mini_arc')
    return load_puzzles(dpath)


def load_arc_puzzles(variant: int = 1, training: bool = True):
    BASE_DIR = Path(__file__).resolve().parent
    dpath = Path(BASE_DIR / ('arc_data_training' if training else 'arc_data_eval'))
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