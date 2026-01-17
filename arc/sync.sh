#!/bin/bash

# --- CONFIGURATION ---
POD_IP="$1"
POD_PORT="$2"
LOCAL_DIR=~/gpu/arcagi/arc
REMOTE_DIR="root@${POD_IP}:/root/arc/"
KEY_PATH="~/.ssh/id_ed25519"
# ---------------------

# Define the RSYNC command with specific includes/excludes
# 1. --include='*/'   : Necessary to let rsync look inside folders.
# 2. --include='*.rs' : The extensions you want.
# 3. --exclude='*'    : Ignore everything that didn't match the rules above.

RSYNC_CMD="rsync -avz \
--include='*/' \
--include='*.py' \
--include='*.sh' \
--include='*.toml' \
--exclude='*' \
-e \"ssh -p ${POD_PORT} -i ${KEY_PATH}\" ${LOCAL_DIR} ${REMOTE_DIR}"

echo "🚀 Starting filtered sync to ${POD_IP}:${POD_PORT}..."

# 1. Run initial sync
eval $RSYNC_CMD

echo "👀 Watching for changes..."

# 2. Watcher loop
fswatch -o ${LOCAL_DIR} | xargs -n1 -I{} bash -c "$RSYNC_CMD"
