#!/bin/bash
ENV_FILE=".env"
TRAIN_S3_PATH="s3://arc-agi/finetune-phase1/train_data_finetune_phase1.tar"
EVAL_S3_PATH="s3://arc-agi/evals/shard_0.jsonl"
BASE_DIR="/home/ubuntu/arc/arc"
TRAIN_DEST_DIR="${BASE_DIR}/arc_data/train_data"
EVAL_DEST_DIR="${BASE_DIR}/arc_data/eval_data"
DATASET_SIZE="10G"
mkdir -p ${TRAIN_DEST_DIR}
mkdir -p ${EVAL_DEST_DIR}

set -a; source .env; set +a
s5cmd cat "$TRAIN_S3_PATH" | pv -s "$DATASET_SIZE" | tar -xf -
mv train_data_phase1/* arc_data/train_data/

s5cmd cp "$EVAL_S3_PATH" "$EVAL_DEST_DIR"


