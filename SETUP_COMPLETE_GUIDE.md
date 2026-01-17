# Complete Setup Guide: Parallel Pose Analysis + GPU VRAM Optimization

## 📋 Table of Contents
1. Prerequisites & System Requirements
2. Step-by-Step Setup (15 minutes)
3. Verification & Testing
4. Daily Workflow
5. Troubleshooting by System
6. Command Reference

---

## 1️⃣ Prerequisites & System Requirements
[text](../../../Downloads/exported-assets/DELIVERY_SUMMARY.md) [text](../../../Downloads/exported-assets/FILE_INDEX_AND_GUIDE.md) [text](../../../Downloads/exported-assets/IMPLEMENTATION_PACKAGE_SUMMARY.md) [text](../../../Downloads/exported-assets/MAKEFILE) [text](../../../Downloads/exported-assets/PARALLEL_IMPL_PLAN.md) [text](../../../Downloads/exported-assets/ROADMAP_VISUAL_GUIDE.md) [text](../../../Downloads/exported-assets/SETUP_COMPLETE_GUIDE.md) [text](../../../Downloads/exported-assets/VALIDATE_ENV_SCRIPT.py)
### A. Hardware Requirements

| Component | RTX 5070 Ti | RTX 4070 Ti | RTX 3070 Ti |
|-----------|-----------|-----------|-----------|
| VRAM | 12GB | 12GB (mobile) | 8GB |
| Compute Cap | 9.0 | 8.9 | 8.6 |
| Driver Min | 550+ | 535+ | 535+ |
| CUDA Support | 12.8 | 12.8 | 12.8 |
| Power | 250W | 115W (mobile) | 290W |

### B. Software Requirements

```bash
# Install Pixi (one-time, any system)
curl -fsSL https://pixi.sh/install.sh | bash

# Verify Pixi
pixi --version
# Should output: pixi 0.X.X or later

# Verify NVIDIA Driver
nvidia-smi
# Should show: CUDA Version: 12.8 or higher

# Verify Python
python --version
# Should be: 3.10+
```

### C. System Preparation

```bash
# Create workspace directory
mkdir -p ~/projects/ai-ml-pipeline
cd ~/projects/ai-ml-pipeline

# Create subdirectories
mkdir -p projects/{pose_analyzer,gpu_optimizer,shared_utils}
mkdir -p data/{pose_references,test_videos,cache}
mkdir -p scripts notebooks config logs
```

---

## 2️⃣ Step-by-Step Setup (15 minutes)

### Step 1: Create Root Pixi Configuration

**File: `pixi.toml`** (in project root)

Copy content from `PIXI_ROOT_CONFIG.toml` (provided separately)

```bash
# Verify structure
ls -la pixi.toml
# Should show the file exists
```

### Step 2: Create Project Configurations

**For Pose Analyzer:**

```bash
# Create directory structure
mkdir -p projects/pose_analyzer/{src/pose_analyzer,api,tests}

# Create pyproject.toml
cat > projects/pose_analyzer/pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pose-analyzer"
version = "0.1.0"
description = "Real-time fitness form detector with multi-stream pose estimation"
requires-python = ">=3.10"
authors = [
    { name = "Your Name", email = "email@example.com" },
]

dependencies = [
    "mediapipe>=0.10.0",
    "ultralytics>=8.5.0",
    "opencv-python>=4.10.0",
    "numpy>=1.26.0",
    "fastapi>=0.115.0",
    "uvicorn>=0.32.0",
    "websockets>=14.2",
    "pydantic>=2.8.0",
    "torch>=2.4.0",
]

[tool.setuptools]
package-dir = { "" = "src" }
EOF

# Create __init__.py
touch projects/pose_analyzer/src/pose_analyzer/__init__.py
```

**For GPU Optimizer:**

```bash
# Create directory structure
mkdir -p projects/gpu_optimizer/{src/gpu_optimizer,api,examples,tests}

# Create pyproject.toml
cat > projects/gpu_optimizer/pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "gpu-optimizer"
version = "0.1.0"
description = "GPU VRAM optimization suite with tensor management"
requires-python = ">=3.10"
authors = [
    { name = "Your Name", email = "email@example.com" },
]

dependencies = [
    "torch>=2.4.0",
    "nvidia-ml-py>=12.535.0",
    "numpy>=1.26.0",
    "fastapi>=0.115.0",
    "prometheus-client>=0.21.0",
    "pydantic>=2.8.0",
]

[tool.setuptools]
package-dir = { "" = "src" }
EOF

# Create __init__.py
touch projects/gpu_optimizer/src/gpu_optimizer/__init__.py
```

### Step 3: Generate Lock File

```bash
# From project root
pixi lock --no-environment

# This creates pixi.lock with all dependencies
# File size should be ~50-100MB (includes PyTorch)
ls -lh pixi.lock
```

### Step 4: Install Projects

```bash
# Install in development mode
pixi run pip install -e projects/pose_analyzer
pixi run pip install -e projects/gpu_optimizer

# Verify installations
pixi run pip list | grep -E "(pose|gpu|torch|mediapipe)"
```

### Step 5: Validate Environment

```bash
# Create validation script
cat > scripts/validate_env.py << 'EOF'
import torch
import sys

print("🔍 Environment Validation")
print("=" * 50)

# PyTorch
print(f"✓ PyTorch: {torch.__version__}")

# CUDA
if torch.cuda.is_available():
    print(f"✓ CUDA Available: True")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  - GPU {i}: {props.name} ({props.total_memory/1e9:.1f}GB)")
else:
    print("⚠️ CUDA not available, using CPU")

# Dependencies
deps = ["mediapipe", "ultralytics", "fastapi", "numpy", "pandas"]
for dep in deps:
    try:
        __import__(dep)
        print(f"✓ {dep}")
    except ImportError:
        print(f"✗ {dep} MISSING")
        sys.exit(1)

print("=" * 50)
print("✅ All validations passed!")
EOF

# Run validation
pixi run python scripts/validate_env.py
```

---

## 3️⃣ Verification & Testing

### Test Pose Analyzer Setup

```python
# In terminal 1: Open Python REPL
pixi run python

# Then in Python:
import mediapipe as mp
from mediapipe.tasks import python as tasks
print("✓ MediaPipe loaded")

import ultralytics
print(f"✓ YOLOv11: {ultralytics.__version__}")

import cv2
print(f"✓ OpenCV: {cv2.__version__}")
```

### Test GPU Optimizer Setup

```python
# In terminal 2: Open Python REPL
pixi run python

# Then in Python:
import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA: {torch.version.cuda}")
print(
    f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB"
)

import pynvml as nvml
nvml.nvmlInit()
print("✓ nvidia-ml-py loaded")
```

### Quick API Test

```bash
# Terminal 1: Start Pose Analyzer API
cd projects/pose_analyzer
pixi run python -m pose_analyzer.api.main --port 8001 &

# Terminal 2: Start GPU Optimizer API
cd projects/gpu_optimizer
pixi run python -m gpu_optimizer.api.main --port 8002 &

# Terminal 3: Test endpoints
sleep 2  # Wait for servers to start
curl http://localhost:8001/health
curl http://localhost:8002/health

# Should both return JSON responses
```

---

## 4️⃣ Daily Workflow

### Terminal Setup (tmux)

```bash
# Create tmux session for development
tmux new-session -d -s ml -x 180 -y 50

# Window 0: Pose Analyzer
tmux new-window -t ml -n pose
tmux send-keys -t ml:pose "cd projects/pose_analyzer && pixi run python -m ipython" Enter

# Window 1: GPU Optimizer
tmux new-window -t ml -n vram
tmux send-keys -t ml:vram "cd projects/gpu_optimizer && pixi run jupyter lab" Enter

# Window 2: GPU Monitoring
tmux new-window -t ml -n monitor
tmux send-keys -t ml:monitor "watch -n 1 nvidia-smi" Enter

# Attach to session
tmux attach -t ml
```

### VSCode Setup

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv",
  "python.formatting.provider": "black",
  "[python]": {
    "editor.formatOnSave": true
  },
  "python.linting.enabled": true
}
```

### Common Commands

```bash
# Development
pixi run python -m ipython              # Interactive Python
pixi run jupyter lab                    # Jupyter Lab
pixi run pytest projects/ -v            # Run tests
pixi run ruff check projects/           # Lint
pixi run ruff format projects/          # Format

# APIs
pixi run --environment cuda python -m pose_analyzer.api.main --port 8001
pixi run --environment cuda python -m gpu_optimizer.api.main --port 8002

# Monitoring
watch -n 1 nvidia-smi
pixi run tensorboard --logdir logs/
```

---

## 5️⃣ Troubleshooting by System

### Issue: "CUDA Version Mismatch"

**Symptoms:** `RuntimeError: CUDA out of memory` or driver errors

**Solution:**

```bash
# Check driver version
nvidia-smi | head -5

# Check PyTorch CUDA version
pixi run python -c "import torch; print(torch.version.cuda)"

# Must show CUDA 12.8 for all systems

# If mismatch, rebuild lock:
pixi lock --no-environment --force
pixi sync --all
```

### Issue: "Not Enough VRAM" on RTX 3070 Ti

**Symptoms:** OOM errors during pose detection with 4 streams

**Solution:**

```python
# In gpu_optimizer: auto-reduce batch size
from gpu_optimizer import auto_batch_size

optimal_batch_size = auto_batch_size(model, max_vram_percent=0.85)
# Returns smaller batch for 8GB GPU
```

### Issue: "Module Not Found" After Git Clone

**Solution:**

```bash
# Reinstall projects
pixi run pip install -e projects/pose_analyzer
pixi run pip install -e projects/gpu_optimizer
```

### Issue: "Permission Denied" on Scripts

**Solution:**

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run with bash instead
bash scripts/quick_start.sh
```

### Issue: Different Paths Between Machines

**Solution:**

Use relative paths or environment variables:

```python
# Good
model_path = Path(__file__).parent / "models" / "pose_landmark.task"

# Better
model_path = Path.home() / ".ml_pipeline" / "models" / "pose_landmark.task"

# Even better
import os
model_path = Path(os.getenv("ML_MODELS_PATH", Path.home() / ".ml_pipeline"))
```

---

## 6️⃣ Command Reference

### Pixi Commands

```bash
# Initialize workspace
pixi init

# Lock dependencies
pixi lock --no-environment

# Run commands in environment
pixi run <command>
pixi run --environment cuda <command>
pixi run --environment cpu <command>

# Add dependencies
pixi add package_name

# Update environment
pixi update

# Remove from cache
pixi cache clean
```

### Project Commands

```bash
# From project directory
cd projects/pose_analyzer

# Install in dev mode
pixi run pip install -e ".[dev]"

# Run tests
pixi run pytest tests/ -v --cov=pose_analyzer

# Format code
pixi run ruff format .
pixi run ruff check . --fix

# Type check
pixi run pyright .
```

### GPU Commands

```bash
# Check GPU status
nvidia-smi

# Monitor in real-time
watch -n 1 nvidia-smi

# Check CUDA info
pixi run python -c "import torch; print(torch.cuda.is_available())"

# Profile GPU memory
pixi run python -c "import torch; torch.cuda.memory._record_memory_history(True)"
```

---

## 📊 Expected Performance Metrics

### Pose Analyzer
- **Single Stream (1080p):** <50ms latency
- **4 Streams Concurrent:** <200ms combined latency
- **GPU Utilization:** 40-60% on RTX 5070 Ti
- **Memory Usage:** 2-3GB VRAM

### GPU Optimizer
- **Memory Profiling:** <100ms per model
- **Batch Size Detection:** 5-10 iterations
- **Gradient Checkpointing Overhead:** 15-20% compute
- **Memory Reduction:** 30-50% with checkpointing

---

## 📞 Support & Resources

### Documentation Files
- `PARALLEL_IMPL_PLAN.md` - Complete implementation timeline
- `PIXI_ROOT_CONFIG.toml` - Root configuration template
- `POSE_ANALYZER_PYPROJECT.toml` - Pose analyzer config
- `GPU_OPTIMIZER_PYPROJECT.toml` - GPU optimizer config

### Official Resources
- Pixi Docs: https://pixi.sh/latest/
- PyTorch Docs: https://pytorch.org/docs/
- MediaPipe: https://developers.google.com/mediapipe
- YOLOv11: https://docs.ultralytics.com/models/yolov11/

### Debugging
```bash
# Verbose output
pixi run --verbose python script.py

# Environment info
pixi info

# Check conflicts
pixi lock --check

# Clean rebuild
pixi sync --force
pixi cache clean
pixi lock --no-environment --force
```

---

**You're now ready to begin development! Start with Week 1 tasks from the main implementation plan.**
