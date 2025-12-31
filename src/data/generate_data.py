from src.data.kant import generate_kant_dataset
from src.data.build_arc_dataset import (
    generate_rearc_data, get_data_dir,
    shard_puzzles, load_rearc_data,
    load_kant_arc
)


def generate_phase1():
    data_dir = get_data_dir()
    generate_rearc_data(rearc_cnt=10)
    kant_outpath = data_dir / 'arc_data_training/kant'
    generate_kant_dataset(n=100, output=kant_outpath)
    puzzles = []
    puzzles += load_rearc_data(num_augs=10)
    puzzles += load_kant_arc()
    shard_puzzles(puzzles)

