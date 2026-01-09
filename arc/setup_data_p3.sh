#!/bin/bash
ENV_FILE=".env"
S3_PATH="s3://arc-agi/finetune-phase3/train_data_phase3.tar"
DEST_DIR="/root/arc_data/train_data"
DATASET_SIZE="100M"

# Checkpoint paths
CHECKPOINT_S3_PATH="s3://arc-agi/finetune-phase2/phase2_checkpoint.pt"
CHECKPOINT_DEST_DIR="/root/checkpoints/phase2"
CHECKPOINT_DEST_FILE="${CHECKPOINT_DEST_DIR}/phase2_checkpoint.pt"

mkdir -p ${DEST_DIR}
mkdir -p ${CHECKPOINT_DEST_DIR}

set -a; source .env; set +a

# Download dataset
echo "Downloading Phase 3 dataset..."
s5cmd cat "$S3_PATH" | pv -s "$DATASET_SIZE" | tar -xf -
mv train_data_phase3/* arc_data/train_data/

# Download Phase 2 checkpoint
echo "Downloading Phase 2 checkpoint..."
s5cmd cp "$CHECKPOINT_S3_PATH" "$CHECKPOINT_DEST_FILE"
echo "Checkpoint saved to: $CHECKPOINT_DEST_FILE"

