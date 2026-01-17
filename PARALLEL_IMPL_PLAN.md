# Parallel Implementation Plan: Pose Analysis + GPU VRAM Optimization

## Overview
This document provides a detailed step-by-step plan for implementing **Project 2 (Pose Analysis)** and **Project 3 (GPU VRAM Optimization)** in parallel using a shared Pixi workspace. Both projects share dependencies but maintain separate codebases for clear separation of concerns.

---

## Architecture: Monorepo with Pixi Workspace

```
ai-ml-pipeline/
├── pixi.toml                          # Root workspace config (CUDA 12.8)
├── pixi.lock                          # Shared lock file for all projects
│
├── projects/
│   ├── pose_analyzer/                 # Project 2: Pose Analysis Pipeline
│   │   ├── pyproject.toml
│   │   ├── src/pose_analyzer/
│   │   │   ├── __init__.py
│   │   │   ├── pose_detector.py       # MediaPipe + YOLOv11 models
│   │   │   ├── biomechanics.py        # Joint angle calculations
│   │   │   ├── form_scorer.py         # Reference-based form analysis
│   │   │   ├── temporal_analyzer.py   # DTW anomaly detection
│   │   │   ├── video_processor.py     # OpenCV stream handling
│   │   │   └── db_connector.py        # TimescaleDB integration
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── main.py                # FastAPI server + WebSocket
│   │   │   ├── routes.py
│   │   │   └── schemas.py
│   │   ├── tests/
│   │   └── README.md
│   │
│   ├── gpu_optimizer/                 # Project 3: GPU VRAM Optimizer
│   │   ├── pyproject.toml
│   │   ├── src/gpu_optimizer/
│   │   │   ├── __init__.py
│   │   │   ├── memory_profiler.py     # Tensor lifecycle tracking
│   │   │   ├── tensor_swapper.py      # GPU ↔ CPU memory management
│   │   │   ├── checkpoint_manager.py  # Gradient checkpointing wrapper
│   │   │   ├── batch_optimizer.py     # Dynamic batch sizing
│   │   │   ├── cuda_analyzer.py       # CUDA memory fragmentation
│   │   │   ├── dashboard.py           # Prometheus metrics export
│   │   │   └── cost_model.py          # Cloud GPU cost estimation
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── main.py                # FastAPI + Prometheus
│   │   │   └── routes.py
│   │   ├── examples/
│   │   │   ├── llama_7b_optimization.py
│   │   │   ├── diffusion_model_tune.py
│   │   │   └── training_pipeline.py
│   │   ├── tests/
│   │   └── README.md
│   │
│   └── shared_utils/                  # Shared utilities (optional)
│       ├── pyproject.toml
│       └── src/shared_utils/
│           ├── config.py
│           ├── logging.py
│           └── data_models.py
│
├── notebooks/                          # Jupyter notebooks for exploration
│   ├── pose_demo.ipynb
│   └── vram_profiling_demo.ipynb
│
├── data/                               # Shared data directory
│   ├── pose_references/               # Reference pose datasets
│   ├── test_videos/                   # Test video files
│   └── cache/
│
├── scripts/                            # Utility scripts
│   ├── setup_databases.sh
│   ├── download_models.py
│   └── benchmark.sh
│
└── docs/                               # Documentation
    ├── SETUP.md
    ├── API.md
    └── TROUBLESHOOTING.md
```

---

## Part 1: Root Pixi Workspace Configuration

### Step 1.1: Initialize Root Workspace

```bash
# Create project directory
mkdir -p ai-ml-pipeline
cd ai-ml-pipeline

# Create directory structure
mkdir -p projects/{pose_analyzer,gpu_optimizer,shared_utils}
mkdir -p data/{pose_references,test_videos,cache}
mkdir -p scripts notebooks config logs
```

### Step 1.2: Create Root `pixi.toml`

**Key Design Decisions:**
- **Workspace structure** (not traditional Pixi project) - allows multiple sub-projects
- **PyPI PyTorch wheels** - better CUDA 12.8 support than conda packages
- **Feature-based environments** - clean separation of CPU vs CUDA dependencies
- **Editable installs** - local packages installed as `path = "projects/..."` with `editable = true`

```toml
# pixi.toml - Root workspace configuration
[workspace]
name = "ai-ml-pipeline"
version = "0.1.0"
description = "Parallel ML projects: Pose Analysis + GPU VRAM Optimization"
authors = ["Your Name <email@example.com>"]
channels = ["conda-forge"]
platforms = ["linux-64"]

# ============================================================================
# Base dependencies (conda-forge, CPU-capable)
# ============================================================================

[dependencies]
python = "3.10.*"

# Computer Vision / Data Science (Conda only)
opencv = "4.10.*"
numpy = "1.26.*"
scipy = "1.14.*"
scikit-learn = "1.5.*"
pandas = "2.2.*"
matplotlib = "3.10.*"

# Time-Series & Anomaly Detection
dtaidistance = "2.3.*"
tslearn = "0.6.*"

# Monitoring / System
psutil = "6.1.*"

# ============================================================================
# PyPI dependencies (PyTorch + ML packages)
# ============================================================================

[pypi-dependencies]
# ML/CV packages (PyPI only)
mediapipe = ">=0.10,<0.11"
kafka-python = ">=2.0,<2.1"
psycopg2-binary = ">=2.9,<3.0"
wandb = ">=0.18,<0.19"
ultralytics = ">=8.4,<8.5"
prometheus-client = ">=0.21,<0.25"
nvidia-ml-py = ">=12.535,<13"

# ============================================================================
# Local workspace packages (editable installs)
# ============================================================================

[pypi-dependencies.pose-analyzer]
path = "projects/pose_analyzer"
editable = true

[pypi-dependencies.gpu-optimizer]
path = "projects/gpu_optimizer"
editable = true

# ============================================================================
# Development Feature
# ============================================================================

[feature.dev.dependencies]
jupyter = "1.0.*"
jupyterlab = "4.2.*"
pytest = "8.2.*"
pytest-cov = "5.*"
pytest-asyncio = "0.24.*"
ruff = "0.8.*"
pyright = "1.1.*"
tensorboard = "2.18.*"

# ============================================================================
# CPU Feature - CPU-only PyTorch from PyPI
# ============================================================================

[feature.cpu-torch]
[feature.cpu-torch.pypi-dependencies]
torch = { version = ">=2.4.0,<2.5", index = "https://download.pytorch.org/whl/cpu" }
torchvision = { version = ">=0.19.0,<0.20", index = "https://download.pytorch.org/whl/cpu" }
torchaudio = { version = ">=2.4.0,<2.5", index = "https://download.pytorch.org/whl/cpu" }

# ============================================================================
# CUDA Feature - CUDA 12.8 PyTorch wheels from PyPI
# ============================================================================

[feature.cuda]
platforms = ["linux-64"]

[feature.cuda.system-requirements]
cuda = "12.0"

[feature.cuda.pypi-dependencies]
# PyTorch 2.7.1 with CUDA 12.8 support (cu128 index)
torch = { version = "==2.7.1", index = "https://download.pytorch.org/whl/cu128" }
torchvision = { version = "==0.22.1", index = "https://download.pytorch.org/whl/cu128" }
torchaudio = { version = "==2.7.1", index = "https://download.pytorch.org/whl/cu128" }

# ============================================================================
# Environments
#   - cpu: base + dev + cpu-torch (CPU-only PyTorch)
#   - cuda: base + dev + cuda (GPU PyTorch CUDA 12.8)
# ============================================================================

[environments]
cpu = { features = ["dev", "cpu-torch"] }
cuda = { features = ["dev", "cuda"] }

# ============================================================================
# Tasks
# ============================================================================

[tasks]
# Code quality
lint = "ruff check projects/ && pyright projects/"
format = "ruff format projects/ && ruff check projects/ --fix"

# Testing & validation
test = "pytest projects/ -v --cov=projects"
validate-env = "python scripts/validate_env.py"
detect-gpu = "python scripts/detect_gpu.py"

# Service APIs
pose-api = { cmd = "python -m pose_analyzer.api.main --port 8001", cwd = "projects/pose_analyzer" }
vram-api = { cmd = "python -m gpu_optimizer.api.main --port 8002", cwd = "projects/gpu_optimizer" }

# Setup & maintenance
download-models = "python scripts/download_models.py"
sync-projects = "bash scripts/sync_projects.sh"

# Clean environment
clean = "rm -rf .pixi/envs && rm pixi.lock"
fresh-install = "pixi install --force-reinstall"
```

### Step 1.3: Initialize Pixi Lock File

```bash
# Generate lock file (creates pixi.lock with all resolved dependencies)
pixi lock --no-environment

# Expected output: pixi.lock (~50-100MB with PyTorch)
ls -lh pixi.lock
```

### Step 1.4: Create `.pixi/env` Management Script

```bash
#!/bin/bash
# scripts/switch_env.sh

ENV_NAME=${1:-cuda}  # Default to cuda

if [ "$ENV_NAME" = "cuda" ]; then
    echo "🚀 Switching to CUDA environment..."
    pixi run --environment cuda python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
elif [ "$ENV_NAME" = "cpu" ]; then
    echo "🚀 Switching to CPU environment..."
    pixi run --environment cpu python -c "import torch; print(f'PyTorch CPU: {torch.__version__}')"
else
    echo "❌ Unknown environment: $ENV_NAME"
    exit 1
fi
```

---

## Part 2: Project Structure Setup

### Step 2.1: Initialize Pose Analyzer Project

```bash
cd ai-ml-pipeline/projects

# Create project structure
mkdir -p pose_analyzer/{src/pose_analyzer,api,tests}
touch pose_analyzer/src/pose_analyzer/__init__.py
```

### Step 2.2: Pose Analyzer `pyproject.toml`

**Key Points:**
- Minimal dependencies (most come from root pixi.toml)
- Uses `package-dir = { "" = "src" }` for src layout
- Version constraints aligned with root configuration

```toml
# projects/pose_analyzer/pyproject.toml
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
    "mediapipe>=0.10,<0.11",
    "ultralytics>=8.4,<8.5",
    "opencv-python>=4.10.0",
    "numpy>=1.26,<1.27",
    "fastapi>=0.115.0",
    "uvicorn>=0.32.0",
    "websockets>=14.2",
    "pydantic>=2.8.0",
    "torch>=2.4.0",
]

[tool.setuptools]
package-dir = { "" = "src" }
```

### Step 2.3: Initialize GPU Optimizer Project

```bash
mkdir -p gpu_optimizer/{src/gpu_optimizer,api,examples,tests}
touch gpu_optimizer/src/gpu_optimizer/__init__.py
```

### Step 2.4: GPU Optimizer `pyproject.toml`

**Key Points:**
- Focused on GPU/VRAM management dependencies
- Includes nvidia-ml-py for GPU monitoring
- Prometheus client for metrics export

```toml
# projects/gpu_optimizer/pyproject.toml
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
    "numpy>=1.26,<1.27",
    "fastapi>=0.115.0",
    "prometheus-client>=0.21,<0.25",
    "pydantic>=2.8.0",
]

[tool.setuptools]
package-dir = { "" = "src" }
```

**Note:** The shared_utils project was created but not yet configured with a pyproject.toml in the actual implementation.

---

## Part 3: Dependency Locking & Environment Setup

### Step 3.1: Generate Lock File

```bash
cd ai-ml-pipeline

# Generate lock file (resolves all dependencies)
pixi lock --no-environment

# Expected output: pixi.lock file (~50-100MB with PyTorch)
ls -lh pixi.lock
```

**What happens:**
- Resolves conda-forge packages (opencv, numpy, scipy, etc.)
- Resolves PyPI packages (mediapipe, ultralytics, etc.)
- Downloads PyTorch wheels from PyPI (cu128 index for CUDA)
- Creates reproducible lock file for all environments

### Step 3.2: Install Projects in Editable Mode

Since the projects are declared in `[pypi-dependencies]` with `editable = true`, they're automatically installed when you use the environment. However, you can verify:

```bash
# Check installed packages
pixi run pip list | grep -E "(pose|gpu|torch)"

# Should show:
# pose-analyzer    0.1.0    /path/to/projects/pose_analyzer
# gpu-optimizer    0.1.0    /path/to/projects/gpu_optimizer
# torch            2.7.1
# torchvision      0.22.1
```

### Step 3.3: Create Validation Script

Create `scripts/validate_env.py`:

```python
#!/usr/bin/env python
"""Validate environment compatibility across systems."""

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
```

### Step 3.4: Run Validation

```bash
# Run validation script
pixi run python scripts/validate_env.py

# Expected output:
# 🔍 Environment Validation
# ==================================================
# ✓ PyTorch: 2.7.1
# ✓ CUDA Available: True
# ✓ CUDA Version: 12.8
# ✓ GPU Count: 1
#   - GPU 0: NVIDIA GeForce RTX 5070 Ti (12.0GB)
# ✓ mediapipe
# ✓ ultralytics
# ✓ fastapi
# ✓ numpy
# ✓ pandas
# ==================================================
# ✅ All validations passed!
```

---

## Part 4: Parallel Development Workflow

### Step 4.1: Terminal Setup for Parallel Development

**Option A: tmux/screen (Recommended)**

```bash
# scripts/dev_session.sh
#!/bin/bash

# Create tmux session with 3 windows
tmux new-session -d -s ml -x 180 -y 50

# Window 0: Pose Analyzer Development
tmux new-window -t ml -n pose
tmux send-keys -t ml:pose "cd projects/pose_analyzer && pixi run python -m ipython" Enter

# Window 1: GPU Optimizer Development  
tmux new-window -t ml -n vram
tmux send-keys -t ml:vram "cd projects/gpu_optimizer && pixi run python -m ipython" Enter

# Window 2: Monitoring & Testing
tmux new-window -t ml -n monitor
tmux send-keys -t ml:monitor "cd ai-ml-pipeline && watch -n 1 nvidia-smi" Enter

# Attach to session
tmux attach -t ml
```

**Option B: VSCode Workspaces**

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "pixi.projects": [
    "${workspaceFolder}/projects/pose_analyzer",
    "${workspaceFolder}/projects/gpu_optimizer"
  ]
}
```

### Step 4.2: Development Workflow Commands

Create `Makefile`:

```makefile
.PHONY: help install test lint format clean run-pose run-vram monitor

help:
	@echo "📚 AI/ML Pipeline Commands"
	@echo "make install        - Install all projects in workspace"
	@echo "make test           - Run all tests"
	@echo "make lint           - Lint all projects"
	@echo "make format         - Format code"
	@echo "make run-pose       - Run pose analyzer API"
	@echo "make run-vram       - Run GPU optimizer API"
	@echo "make monitor        - Monitor GPU/CPU usage"
	@echo "make clean          - Clean build artifacts"

install:
	pixi lock --no-environment
	pixi run pip install -e projects/shared_utils
	pixi run pip install -e projects/pose_analyzer
	pixi run pip install -e projects/gpu_optimizer

test:
	pixi run pytest projects/ -v --cov=projects --cov-report=html

lint:
	pixi run ruff check projects/
	pixi run pyright projects/

format:
	pixi run ruff format projects/
	pixi run ruff check projects/ --fix

run-pose:
	pixi run --environment cuda python -m pose_analyzer.api.main

run-vram:
	pixi run --environment cuda python -m gpu_optimizer.api.main

monitor:
	watch -n 1 nvidia-smi

clean:
	find projects -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
	find projects -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null
	find projects -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null
	find projects -type f -name "*.pyc" -delete
	find projects -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null
```

---

## Part 5: Implementation Timeline (6 Weeks)

### **Week 1: Foundation & Setup**

#### Days 1-2: Environment & Workspace
- [ ] Clone/initialize Pixi workspace
- [ ] Validate CUDA 12.8 on all 3 systems (5070 Ti, 4070 Ti, 3070 Ti)
- [ ] Run `validate_env.py` on each machine
- [ ] Set up development terminals (tmux/VSCode)

#### Days 3-4: Pose Analyzer - Data Collection
- [ ] Download sample fitness videos (squats, push-ups, lunges)
- [ ] Collect reference dataset (correct form examples)
- [ ] Store in `data/pose_references/`
- [ ] Implement basic video loader with OpenCV

#### Days 5-7: GPU Optimizer - Memory Profiling Baseline
- [ ] Create `memory_profiler.py` with `nvidia-ml-py`
- [ ] Baseline memory tracking for ResNet, BERT, Llama-7B
- [ ] Create Prometheus metrics export
- [ ] Build simple dashboard in Grafana

---

### **Week 2: Core Implementations**

#### Days 8-10: Pose Analyzer - Detection Pipeline
- [ ] Implement MediaPipe pose detector
- [ ] Implement YOLOv11 pose detector (compare both)
- [ ] Extract 17 COCO keypoints
- [ ] Process 4 concurrent 1080p streams (test single GPU)

#### Days 11-14: GPU Optimizer - Tensor Management
- [ ] Implement tensor swapper (GPU ↔ CPU)
- [ ] Implement gradient checkpointing wrapper
- [ ] Dynamic batch sizing algorithm
- [ ] Apply to Llama-7B training (first pass)

---

### **Week 3: Biomechanics & Advanced Features**

#### Days 15-17: Pose Analyzer - Angle Calculations
- [ ] Implement joint angle calculations (shoulder, elbow, hip, knee)
- [ ] Build reference-based form scoring
- [ ] Implement DTW (Dynamic Time Warping) anomaly detection
- [ ] Real-time temporal analysis

#### Days 18-21: GPU Optimizer - Advanced Profiling
- [ ] CUDA memory fragmentation analysis
- [ ] Cost modeling for cloud GPU instances
- [ ] Gradient accumulation integration
- [ ] Mixed precision (fp16/bf16) support

---

### **Week 4: API & Integration**

#### Days 22-24: Pose Analyzer - FastAPI Server
- [ ] Build FastAPI routes (upload video, real-time webcam)
- [ ] WebSocket for real-time feedback
- [ ] Generate audio/visual alerts for form deviation
- [ ] TimescaleDB integration for analytics

#### Days 25-28: GPU Optimizer - Dashboard & API
- [ ] FastAPI monitoring endpoints
- [ ] Prometheus metrics collection
- [ ] Grafana dashboard setup
- [ ] CLI tool for model optimization

---

### **Week 5: Cross-System Testing**

#### Days 29-31: Multi-GPU Testing
- [ ] Test pose analyzer on 5070 Ti (main dev)
- [ ] Test pose analyzer on 4070 Ti (training machine)
- [ ] Test pose analyzer on 3070 Ti (secondary)
- [ ] Benchmark latency & memory across systems

#### Days 32-35: VRAM Optimization Validation
- [ ] Apply to GeriApp training pipeline
- [ ] Apply to Pumpl model training
- [ ] Measure memory reduction %
- [ ] Document cross-system compatibility

---

### **Week 6: Production & Documentation**

#### Days 36-38: Optimization & Hardening
- [ ] Performance profiling & tuning
- [ ] Error handling & edge cases
- [ ] Docker containerization (optional)
- [ ] Security review

#### Days 39-42: Documentation
- [ ] API documentation
- [ ] Setup guide for multiple machines
- [ ] Troubleshooting guide
- [ ] Performance benchmarks
- [ ] Project reports & lessons learned

---

## Part 6: Cross-System Configuration

### Step 6.1: Machine-Specific Configurations

Create `config/machines.yml`:

```yaml
# config/machines.yml
machines:
  desktop-5070ti:
    name: "Main Dev Machine"
    gpu: "RTX 5070 Ti"
    vram_gb: 12
    cuda_sm: "90"
    primary: true
    use_for: ["development", "inference", "testing"]
    environment: cuda

  laptop-4070ti:
    name: "Training Laptop"
    gpu: "RTX 4070 Ti Mobile"
    vram_gb: 12
    cuda_sm: "89"
    primary: false
    use_for: ["training", "optimization"]
    environment: cuda
    power_limit: 90  # TDP limit for laptop

  pc-3070ti:
    name: "Secondary PC"
    gpu: "RTX 3070 Ti"
    vram_gb: 8
    cuda_sm: "86"
    primary: false
    use_for: ["backup", "inference", "testing"]
    environment: cuda
    batch_size_limit: 32  # More constrained
```

### Step 6.2: Automatic GPU Detection Script

Create `scripts/detect_gpu.py`:

```python
#!/usr/bin/env python
"""Detect GPU and auto-configure Pixi environment."""

import torch
import json
from pathlib import Path

def detect_and_configure():
    """Detect GPU and create machine-specific config."""
    
    if not torch.cuda.is_available():
        print("⚠️  No CUDA GPU detected. Using CPU mode.")
        return "cpu"
    
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    compute_capability = torch.cuda.get_device_capability(0)
    
    print(f"🎮 GPU Detected: {gpu_name}")
    print(f"💾 VRAM: {total_memory:.1f}GB")
    print(f"🔧 Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
    
    config = {
        "gpu": gpu_name,
        "vram_gb": total_memory,
        "compute_capability": f"{compute_capability[0]}.{compute_capability[1]}",
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    
    # Save to config
    config_path = Path("~/.ml_pipeline_gpu.json").expanduser()
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Config saved to {config_path}")
    return "cuda"

if __name__ == "__main__":
    detect_and_configure()
```

### Step 6.3: Network Sync Script (For Multi-Machine Development)

Create `scripts/sync_projects.sh`:

```bash
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
```

---

## Part 7: Pre-Packaged Frameworks & Libraries

### Step 7.1: Pose Detection Frameworks

**Option A: MediaPipe (Recommended for beginners)**
- ✅ Lightweight, real-time
- ✅ Built-in pose landmark model
- ❌ Less customizable

```python
# projects/pose_analyzer/src/pose_analyzer/pose_detector.py
import mediapipe as mp
from mediapipe.tasks import python as task_python
from mediapipe.tasks.python import vision

BaseOptions = task_python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

class MediaPipePoseDetector:
    def __init__(self, model_path: str = None):
        """Initialize MediaPipe pose landmarker."""
        if model_path is None:
            # Auto-download from MediaPipe models
            model_path = "path/to/pose_landmarker_full.task"
        
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_poses=2,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        self.landmarker = PoseLandmarker.create_from_options(options)
    
    def detect(self, frame):
        """Detect pose landmarks in frame."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self.landmarker.detect(mp_image)
        return result.pose_landmarks
```

**Option B: YOLOv11 (Recommended for production)**
- ✅ Faster, more accurate, multi-person support
- ✅ Easier training/fine-tuning
- ❌ Requires GPU for speed

```python
# projects/pose_analyzer/src/pose_analyzer/yolo_detector.py
from ultralytics import YOLO
import cv2

class YOLOPoseDetector:
    def __init__(self, model_variant: str = "n"):  # n, s, m, l, x
        """Initialize YOLOv11 pose detector."""
        self.model = YOLO(f"yolov11{model_variant}-pose.pt")
    
    def detect(self, frame):
        """Detect pose keypoints using YOLOv11."""
        results = self.model(frame, conf=0.5, iou=0.45)
        return results[0].keypoints
```

### Step 7.2: Memory Optimization Frameworks

**Pre-packaged optimization modules:**

```python
# projects/gpu_optimizer/src/gpu_optimizer/memory_profiler.py
import torch
import nvidia_ml_py as ml

class MemoryProfiler:
    """Track tensor memory usage across layers."""
    
    def __init__(self):
        self.snapshots = []
    
    def profile_forward_pass(self, model, input_tensor):
        """Capture memory during forward pass."""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            output = model(input_tensor)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        return output, peak_memory, prof
```

### Step 7.3: Pre-trained Model Downloads

Create `scripts/download_models.py`:

```python
#!/usr/bin/env python
"""Download pre-trained models for projects."""

import os
from pathlib import Path
from ultralytics import YOLO
import mediapipe as mp

def setup_models():
    """Download and cache models."""
    
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 Downloading pose estimation models...")
    
    # YOLOv11 variants
    for variant in ['n', 's', 'm']:
        print(f"  - YOLOv11{variant}-pose.pt...")
        YOLO(f"yolov11{variant}-pose.pt")  # Auto-downloads to ~/.yolo/runs/
    
    print("✅ Models downloaded successfully!")
    print(f"📁 Models saved to: {model_dir}")

if __name__ == "__main__":
    setup_models()
```

---

## Part 8: Quick Start Guide

### Step 8.1: First-Time Setup (All Systems)

```bash
# 1. Clone/initialize
git clone <repo> ai-ml-pipeline
cd ai-ml-pipeline

# 2. Initialize Pixi
pixi init --workspace

# 3. Install all projects
make install

# 4. Validate environment
pixi run python scripts/detect_gpu.py

# 5. Download models
pixi run python scripts/download_models.py

# 6. Run tests
make test
```

### Step 8.2: Daily Development

```bash
# Terminal 1: Pose Analyzer Development
cd projects/pose_analyzer
pixi run python -m ipython
# In ipython:
from src.pose_analyzer import MediaPipePoseDetector
detector = MediaPipePoseDetector()

# Terminal 2: GPU Optimizer Development
cd projects/gpu_optimizer
pixi run jupyter lab

# Terminal 3: Monitor GPU
watch -n 1 nvidia-smi
```

### Step 8.3: Running APIs in Parallel

```bash
# Terminal 1: Pose Analyzer API
pixi run --environment cuda python -m pose_analyzer.api.main --port 8001

# Terminal 2: GPU Optimizer API
pixi run --environment cuda python -m gpu_optimizer.api.main --port 8002

# Check both running:
curl http://localhost:8001/health
curl http://localhost:8002/health
```

---

## Part 9: Troubleshooting Multi-System Setup

### Issue: CUDA Mismatch Across Systems

```bash
# Solution: Use Pixi's CUDA override
pixi run --environment cuda CUDA_DEVICE_ORDER=PCI_BUS_ID python script.py
```

### Issue: Different GPU VRAM Sizes

```python
# Automatic batch size detection
from gpu_optimizer import auto_batch_size

optimal_batch_size = auto_batch_size(model, max_vram_percent=0.9)
```

### Issue: Model Weights Not Loading

```bash
# Clear cache and re-download
rm -rf ~/.yolo/
pixi run python scripts/download_models.py
```

---

## Part 10: Monitoring & Metrics

### Setup Prometheus + Grafana (Optional but Recommended)

```bash
# Start Prometheus
pixi run prometheus --config.file=config/prometheus.yml

# Start Grafana
pixi run grafana-server

# Dashboards available at:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

---

## Success Checklist

- [ ] Pixi workspace initialized with CUDA 12.8
- [ ] Both projects created with correct structure
- [ ] `pixi lock` generates single lock file for all projects
- [ ] GPU detection works on all 3 machines
- [ ] Pose analyzer detects keypoints in real-time
- [ ] GPU optimizer profiles memory usage
- [ ] Both APIs run simultaneously without conflicts
- [ ] Cross-system sync script tested
- [ ] Documentation complete
- [ ] Unit tests passing (>80% coverage)
- [ ] Performance benchmarks established

---

**Next Steps:**
1. Run Step 1.1-1.4 (Pixi setup)
2. Run Step 2.1-2.5 (Project initialization)
3. Run Step 3.1-3.2 (Dependency locking)
4. Begin Week 1 implementation
