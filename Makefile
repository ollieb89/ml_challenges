# Makefile for AI/ML Pipeline Development
# Usage: make <target>
# Run `make help` for all available commands

.PHONY: help install test lint format clean run-pose run-vram monitor \
        validate validate-cuda download-models sync-projects dev-session

# Default target
help:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║       AI/ML Pipeline - Development Command Reference          ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  make install              Install all projects (pose + vram)"
	@echo "  make download-models      Download pre-trained models"
	@echo "  make validate             Validate environment setup"
	@echo "  make validate-cuda        Check CUDA 12.8 compatibility"
	@echo ""
	@echo "🚀 Development & Running:"
	@echo "  make run-pose             Start Pose Analyzer API (port 8001)"
	@echo "  make run-vram             Start GPU Optimizer API (port 8002)"
	@echo "  make dev-session          Create tmux session for development"
	@echo "  make monitor              Monitor GPU/CPU in real-time"
	@echo ""
	@echo "🧪 Testing & Quality:"
	@echo "  make test                 Run all tests with coverage"
	@echo "  make lint                 Check code style (ruff + pyright)"
	@echo "  make format               Format code (ruff)"
	@echo ""
	@echo "🔧 Utilities:"
	@echo "  make sync-projects        Sync to laptop/secondary PC (requires config)"
	@echo "  make clean                Clean build artifacts"
	@echo "  make clean-all            Deep clean (cache + lock)"
	@echo ""
	@echo "📊 Information:"
	@echo "  make info                 Display environment info"
	@echo "  make list-gpus            Show available GPUs"
	@echo "  make project-tree         Display project structure"
	@echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Setup & Installation
# ═══════════════════════════════════════════════════════════════════════════

install:
	@echo "📦 Installing all projects..."
	pixi lock --no-environment
	pixi run pip install -e projects/shared_utils 2>/dev/null || true
	pixi run pip install -e projects/pose_analyzer
	pixi run pip install -e projects/gpu_optimizer
	@echo "✅ Installation complete!"

download-models:
	@echo "📥 Downloading pre-trained models..."
	pixi run python scripts/download_models.py
	@echo "✅ Models downloaded!"

validate:
	@echo "🔍 Validating environment..."
	pixi run python scripts/validate_env.py

validate-cuda:
	@echo "🔍 Checking CUDA 12.8 compatibility..."
	pixi run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'Available: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'No GPU')"
	
# ═══════════════════════════════════════════════════════════════════════════
# Development & Running
# ═══════════════════════════════════════════════════════════════════════════

run-pose:
	@echo "🎬 Starting Pose Analyzer API (port 8001)..."
	@echo "📍 Access at: http://localhost:8001"
	pixi run --environment cuda python -m pose_analyzer.api.main --port 8001

run-vram:
	@echo "⚙️  Starting GPU Optimizer API (port 8002)..."
	@echo "📍 Access at: http://localhost:8002"
	pixi run --environment cuda python -m gpu_optimizer.api.main --port 8002

dev-session:
	@echo "🚀 Creating tmux development session..."
	tmux new-session -d -s ml -x 180 -y 50 -c "$(PWD)"
	tmux new-window -t ml -n pose -c "$(PWD)/projects/pose_analyzer"
	tmux send-keys -t ml:pose "pixi run python -m ipython" Enter
	tmux new-window -t ml -n vram -c "$(PWD)/projects/gpu_optimizer"
	tmux send-keys -t ml:vram "pixi run jupyter lab" Enter
	tmux new-window -t ml -n monitor
	tmux send-keys -t ml:monitor "watch -n 1 nvidia-smi" Enter
	tmux attach -t ml

monitor:
	@echo "📊 Monitoring GPU/CPU (Ctrl+C to stop)..."
	watch -n 1 nvidia-smi

# ═══════════════════════════════════════════════════════════════════════════
# Testing & Quality
# ═══════════════════════════════════════════════════════════════════════════

test:
	@echo "🧪 Running tests with coverage..."
	pixi run pytest projects/ -v --cov=projects --cov-report=html
	@echo "📊 Coverage report: htmlcov/index.html"

test-pose:
	@echo "🧪 Testing Pose Analyzer..."
	pixi run pytest projects/pose_analyzer/tests -v

test-vram:
	@echo "🧪 Testing GPU Optimizer..."
	pixi run pytest projects/gpu_optimizer/tests -v

lint:
	@echo "🔍 Linting code (ruff + pyright)..."
	pixi run ruff check projects/
	pixi run pyright projects/
	@echo "✅ Linting complete!"

format:
	@echo "📝 Formatting code..."
	pixi run ruff format projects/
	pixi run ruff check projects/ --fix
	@echo "✅ Formatting complete!"

# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

sync-projects:
	@echo "🔄 Syncing projects to remote..."
	@echo "Usage: make sync-projects REMOTE=user@host"
	@if [ -z "$(REMOTE)" ]; then \
		echo "❌ REMOTE not set. Example: make sync-projects REMOTE=user@laptop"; \
		exit 1; \
	fi
	@bash scripts/sync_projects.sh $(REMOTE)

clean:
	@echo "🧹 Cleaning build artifacts..."
	find projects -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find projects -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find projects -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find projects -type f -name "*.pyc" -delete
	find projects -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	find projects -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	@echo "✅ Clean complete!"

clean-all: clean
	@echo "🧹 Deep clean (removing lock file and cache)..."
	rm -f pixi.lock
	pixi cache clean
	@echo "✅ Deep clean complete!"
	@echo "⚠️  Run 'make install' to regenerate lock file"

# ═══════════════════════════════════════════════════════════════════════════
# Information & Debugging
# ═══════════════════════════════════════════════════════════════════════════

info:
	@echo "📋 Environment Information:"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Pixi Version: $$(pixi --version)"
	@echo "Python Version: $$(python --version)"
	@echo "Current Directory: $$(pwd)"
	@echo ""
	@echo "NVIDIA Driver:"
	@nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | xargs echo "  Version:"
	@echo ""
	@echo "Pixi Environments:"
	@echo "  - cuda (GPU)"
	@echo "  - cpu (Fallback)"
	@echo ""
	@echo "Lock File Status:"
	@if [ -f pixi.lock ]; then \
		echo "  ✓ pixi.lock exists ($$(wc -l < pixi.lock | tr -d ' ') lines)"; \
	else \
		echo "  ✗ pixi.lock missing - run 'make install'"; \
	fi

list-gpus:
	@echo "🎮 Available GPUs:"
	@nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

project-tree:
	@echo "📁 Project Structure:"
	@tree -L 3 -I '__pycache__|.pytest_cache|*.egg-info' projects/ 2>/dev/null || find projects -type d | head -20

# ═══════════════════════════════════════════════════════════════════════════
# Development Shortcuts
# ═══════════════════════════════════════════════════════════════════════════

# Quick shortcuts for frequent tasks
shell-cuda:
	pixi run --environment cuda python -m ipython

shell-cpu:
	pixi run --environment cpu python -m ipython

notebook:
	pixi run jupyter lab

tensorboard:
	pixi run tensorboard --logdir logs/

# ═══════════════════════════════════════════════════════════════════════════
# Parallel Development (Advanced)
# ═══════════════════════════════════════════════════════════════════════════

# Run both APIs with automatic port management
run-all:
	@echo "🚀 Starting all services..."
	@echo "  📡 Pose Analyzer: http://localhost:8001"
	@echo "  ⚙️  GPU Optimizer: http://localhost:8002"
	@echo "  📊 GPU Monitor: watch -n 1 nvidia-smi"
	@echo ""
	@echo "Launching in background..."
	@pixi run --environment cuda python -m pose_analyzer.api.main --port 8001 > logs/pose_api.log 2>&1 &
	@pixi run --environment cuda python -m gpu_optimizer.api.main --port 8002 > logs/gpu_api.log 2>&1 &
	@sleep 2
	@echo "✅ Both APIs running in background"
	@echo "📋 Logs: tail -f logs/*.log"

stop-all:
	@echo "🛑 Stopping all services..."
	@pkill -f "pose_analyzer.api.main" || echo "  (Pose API not running)"
	@pkill -f "gpu_optimizer.api.main" || echo "  (GPU API not running)"
	@echo "✅ Services stopped"

# ═══════════════════════════════════════════════════════════════════════════
# Default Targets
# ═══════════════════════════════════════════════════════════════════════════

.DEFAULT_GOAL := help

# Ensure target directories exist
$(shell mkdir -p logs)

# Run tests on file changes (requires entr: brew install entr)
watch-tests:
	@echo "👀 Watching for changes and running tests..."
	@find projects -name "*.py" | entr -c make test

# Format on save
watch-format:
	@echo "👀 Watching for changes and formatting..."
	@find projects -name "*.py" | entr -c make format
