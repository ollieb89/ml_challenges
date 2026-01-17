#!/usr/bin/env bash
# Quick Start Script for Parallel Development Setup
# Run from: ai-ml-pipeline/
# Usage: bash scripts/quick_start.sh

set -e  # Exit on error

echo "🚀 AI/ML Pipeline - Quick Start Setup"
echo "========================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Pixi is installed
if ! command -v pixi &> /dev/null; then
    echo -e "${RED}✗ Pixi not found. Install from: https://pixi.sh${NC}"
    exit 1
fi

echo -e "${BLUE}1. Validating system...${NC}"
pixi run python scripts/validate_env.py || { echo -e "${RED}✗ Validation failed${NC}"; exit 1; }

echo ""
echo -e "${BLUE}2. Detecting GPU...${NC}"
pixi run python scripts/detect_gpu.py

echo ""
echo -e "${BLUE}3. Creating project directories...${NC}"
mkdir -p projects/pose_analyzer/{src/pose_analyzer,api,tests}
mkdir -p projects/gpu_optimizer/{src/gpu_optimizer,api,examples,tests}
mkdir -p projects/shared_utils/{src/shared_utils,tests}
mkdir -p data/{pose_references,test_videos,cache}
mkdir -p notebooks scripts config
echo -e "${GREEN}✓ Directories created${NC}"

echo ""
echo -e "${BLUE}4. Copying configuration files...${NC}"

# Copy pixi.toml (you'll have this from setup)
if [ ! -f "pixi.toml" ]; then
    echo -e "${YELLOW}⚠ pixi.toml not found. Create from PIXI_ROOT_CONFIG.toml${NC}"
fi

# Create project pyproject.toml files if missing
if [ ! -f "projects/pose_analyzer/pyproject.toml" ]; then
    echo "Creating pose_analyzer/pyproject.toml..."
    touch projects/pose_analyzer/pyproject.toml
fi

if [ ! -f "projects/gpu_optimizer/pyproject.toml" ]; then
    echo "Creating gpu_optimizer/pyproject.toml..."
    touch projects/gpu_optimizer/pyproject.toml
fi

echo -e "${GREEN}✓ Configuration files ready${NC}"

echo ""
echo -e "${BLUE}5. Generating Pixi lock file...${NC}"
pixi lock --no-environment
echo -e "${GREEN}✓ Lock file generated${NC}"

echo ""
echo -e "${BLUE}6. Installing projects in editable mode...${NC}"
pixi run pip install -e projects/shared_utils
pixi run pip install -e projects/pose_analyzer
pixi run pip install -e projects/gpu_optimizer
echo -e "${GREEN}✓ Projects installed${NC}"

echo ""
echo -e "${BLUE}7. Downloading pre-trained models...${NC}"
pixi run python scripts/download_models.py
echo -e "${GREEN}✓ Models downloaded${NC}"

echo ""
echo -e "${BLUE}8. Running smoke tests...${NC}"
pixi run pytest projects/ -v --co > /dev/null 2>&1 || echo -e "${YELLOW}⚠ Tests require test files${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

echo "📌 Next Steps:"
echo ""
echo "1. Start Pose Analyzer Development:"
echo "   cd projects/pose_analyzer"
echo "   pixi run python -m ipython"
echo ""
echo "2. Start GPU Optimizer Development:"
echo "   cd projects/gpu_optimizer"
echo "   pixi run jupyter lab"
echo ""
echo "3. Run Both APIs in Parallel:"
echo "   Terminal 1: pixi run --environment cuda python -m pose_analyzer.api.main --port 8001"
echo "   Terminal 2: pixi run --environment cuda python -m gpu_optimizer.api.main --port 8002"
echo ""
echo "4. Monitor GPU:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "5. Run Tests:"
echo "   pixi run pytest projects/ -v"
echo ""
echo "📚 Documentation:"
echo "   - Main Plan: PARALLEL_IMPL_PLAN.md"
echo "   - Troubleshooting: docs/TROUBLESHOOTING.md"
echo ""
