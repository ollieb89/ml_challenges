#!/bin/bash
# Quick data collection script for Day 1 Evening task
# Downloads fitness videos and extracts frames at 30fps
# 
# Usage: pixi exec bash scripts/quick_collect_data.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=================================="
echo "Fitness Data Collection - Day 1"
echo "=================================="
echo ""
echo "✓ Running in pixi environment"
echo "✓ Dependencies managed by pixi"
echo ""

# Run the collection script
echo "Starting data collection..."
echo "  - Videos: 15 fitness videos (squats, push-ups, deadlifts)"
echo "  - Output: data/pose_references/"
echo "  - FPS: 30 frames per second"
echo ""

python scripts/collect_fitness_data.py \
    --videos data/pose_references/video_list.json \
    --output data/pose_references \
    --fps 30

echo ""
echo "=================================="
echo "Data Collection Complete!"
echo "=================================="
echo ""
echo "Summary:"
echo "  - Videos: data/pose_references/videos/"
echo "  - Frames: data/pose_references/frames/"
echo "  - Metadata: data/pose_references/metadata.csv"
echo ""
echo "Next steps:"
echo "  1. Review extracted frames"
echo "  2. Verify metadata.csv"
echo "  3. Continue with Day 1 Afternoon task (Memory Tracer)"
echo ""
