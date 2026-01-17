#!/bin/bash
ENV_FILE=".env"
BASE_PATH="/home/ubuntu/arc/arc"
ARC1_EVALS="s3://arc-agi/arc1-evals/shard_0.jsonl"
ARC2_EVALS="s3://arc-agi/arc2-evals/shard_0.jsonl"
DEST_DIR="${BASE_PATH}/arc_data/eval_data"
ARC1_DEST_DIR="${BASE_PATH}/arc_data/eval_data/arc1"
ARC2_DEST_DIR="${BASE_PATH}/arc_data/eval_data/arc2"
DATASET_SIZE="100M"

# Checkpoint paths
CHECKPOINT_S3_PATH="s3://arc-agi/finetune-phase2/phase2.pt"
LORA_CHECKPOINT_S3="s3://arc-agi/finetune-phase3/lora.pt"
CHECKPOINT_DEST_DIR="${BASE_PATH}/checkpoints"
CHECKPOINT_DEST_FILE="${CHECKPOINT_DEST_DIR}/phase2_checkpoint.pt"
LORA_DEST_FILE="${CHECKPOINT_DEST_DIR}/lora.pt"

mkdir -p ${ARC1_DEST_DIR}
mkdir -p ${ARC2_DEST_DIR}
mkdir -p ${CHECKPOINT_DEST_DIR}

set -a; source .env; set +a

# Download dataset
s5cmd cp "$ARC1_EVALS" "$ARC1_DEST_DIR"
s5cmd cp "$ARC2_EVALS" "$ARC2_DEST_DIR"

# Download Phase 1 checkpoint
echo "Downloading Phase 2 checkpoint..."
s5cmd cp "$CHECKPOINT_S3_PATH" "$CHECKPOINT_DEST_FILE"
echo "Checkpoint saved to: $CHECKPOINT_DEST_FILE"
echo "Downloading lora checkpoint..."
s5cmd cp "$LORA_CHECKPOINT_S3" "$LORA_DEST_FILE"

