#!/bin/bash
# Sync projects between machines via SSH

REMOTE_HOST="${1:-user@laptop-ip}"
PROJECT_PATH="~/ai-ml-pipeline/projects"

echo "🔄 Syncing projects..."

# Sync pose_analyzer
rsync -avz --delete \
  ${PROJECT_PATH}/pose_analyzer/ \
  ${REMOTE_HOST}:${PROJECT_PATH}/pose_analyzer/

# Sync gpu_optimizer
rsync -avz --delete \
  ${PROJECT_PATH}/gpu_optimizer/ \
  ${REMOTE_HOST}:${PROJECT_PATH}/gpu_optimizer/

# Sync shared data
rsync -avz --delete \
  ${PROJECT_PATH}/data/ \
  ${REMOTE_HOST}:${PROJECT_PATH}/data/

echo "✅ Sync complete!"
