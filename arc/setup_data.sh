#!/bin/bash
ENV_FILE=".env"
S3_PATH="s3://arc-agi/finetune-phase1/train_data_finetune_phase1.tar"
DEST_DIR="/root/arc_data/train_data"
DATASET_SIZE="25G"
mkdir -p ${DEST_DIR}
set -a; source .env; set +a
s5cmd cat "$S3_PATH" | pv -s "$DATASET_SIZE" | tar -xf -
mv train_data_phase1/* arc_data/train_data/
