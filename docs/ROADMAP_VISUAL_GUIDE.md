# 🗺️ Visual Implementation Roadmap

## File Organization & Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│              START HERE: Read These Files First                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 1. IMPLEMENTATION_PACKAGE_SUMMARY.md                           │
│    └─ What you got, quick start, overview                     │
│                                                                 │
│ 2. SETUP_COMPLETE_GUIDE.md  ← READ THIS SECOND                 │
│    └─ Step-by-step setup (15 minutes)                          │
│                                                                 │
│ 3. PARALLEL_IMPL_PLAN.md                                       │
│    └─ Deep-dive architecture & 6-week timeline                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│         COPY THESE FILES TO PROJECT ROOT                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Pixi Root Configuration:                                       │
│   PIXI_ROOT_CONFIG.toml                                        │
│   └─ cp to: ai-ml-pipeline/pixi.toml                           │
│                                                                 │
│ Project Configurations:                                        │
│   POSE_ANALYZER_PYPROJECT.toml                                 │
│   └─ cp to: projects/pose_analyzer/pyproject.toml              │
│                                                                 │
│   GPU_OPTIMIZER_PYPROJECT.toml                                 │
│   └─ cp to: projects/gpu_optimizer/pyproject.toml              │
│                                                                 │
│ Scripts:                                                        │
│   QUICK_START_SCRIPT.sh                                        │
│   └─ cp to: scripts/quick_start.sh → chmod +x                  │
│                                                                 │
│   VALIDATE_ENV_SCRIPT.py                                       │
│   └─ cp to: scripts/validate_env.py                            │
│                                                                 │
│ Build Tool:                                                    │
│   MAKEFILE                                                     │
│   └─ cp to: ai-ml-pipeline/Makefile                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│         EXECUTION TIMELINE                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ T+0 min:  Read IMPLEMENTATION_PACKAGE_SUMMARY.md               │
│                                                                 │
│ T+5 min:  Create project root                                  │
│           mkdir -p ai-ml-pipeline                              │
│                                                                 │
│ T+10 min: Copy configuration files                             │
│           (6 .toml/.sh/.py files)                              │
│                                                                 │
│ T+15 min: Follow SETUP_COMPLETE_GUIDE.md section 2             │
│           (Step 1-5 setup)                                     │
│                                                                 │
│ T+20 min: Run validation                                       │
│           make validate                                        │
│                                                                 │
│ T+25 min: Start development                                    │
│           make dev-session                                     │
│                                                                 │
│ T+30 min: Begin Week 1 tasks from PARALLEL_IMPL_PLAN.md        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Command Quick Reference

### 🚀 First Time Setup

```bash
# 1. Copy all files to project root
cd ai-ml-pipeline
cp /path/to/PIXI_ROOT_CONFIG.toml pixi.toml
cp /path/to/POSE_ANALYZER_PYPROJECT.toml projects/pose_analyzer/pyproject.toml
cp /path/to/GPU_OPTIMIZER_PYPROJECT.toml projects/gpu_optimizer/pyproject.toml
cp /path/to/QUICK_START_SCRIPT.sh scripts/quick_start.sh
cp /path/to/VALIDATE_ENV_SCRIPT.py scripts/validate_env.py
cp /path/to/MAKEFILE ./Makefile
chmod +x scripts/quick_start.sh

# 2. Run automated setup
bash scripts/quick_start.sh

# 3. Verify everything
make validate
```

### 📦 Daily Development

```bash
# Check environment
make validate-cuda

# Download models (one-time)
make download-models

# Start development session (tmux with 3 terminals)
make dev-session

# In terminal within tmux:
#   - Window "pose":    cd projects/pose_analyzer && python
#   - Window "vram":    cd projects/gpu_optimizer && jupyter lab
#   - Window "monitor": watch -n 1 nvidia-smi
```

### 🧪 Testing & Quality

```bash
# Run all tests
make test

# Lint code
make lint

# Format code
make format

# Combined (lint + format + test)
make lint && make format && make test
```

### ⚙️ Running APIs

```bash
# Terminal 1: Start Pose Analyzer
make run-pose
# Access: http://localhost:8001

# Terminal 2: Start GPU Optimizer
make run-vram
# Access: http://localhost:8002

# OR run both in background:
make run-all
make stop-all
```

---

## 🔄 File Dependencies & Flow

```
pixi.toml (ROOT CONFIG)
    │
    ├─→ projects/pose_analyzer/pyproject.toml
    │   ├─→ src/pose_analyzer/
    │   │   ├─ pose_detector.py (MediaPipe/YOLOv11)
    │   │   ├─ biomechanics.py
    │   │   ├─ form_scorer.py
    │   │   └─ ...
    │   └─→ api/main.py (FastAPI)
    │
    ├─→ projects/gpu_optimizer/pyproject.toml
    │   ├─→ src/gpu_optimizer/
    │   │   ├─ memory_profiler.py
    │   │   ├─ tensor_swapper.py
    │   │   ├─ checkpoint_manager.py
    │   │   └─ ...
    │   └─→ api/main.py (FastAPI + Prometheus)
    │
    └─→ pixi.lock (SHARED - ensures consistency!)
        └─→ Contains ALL dependencies
            - PyTorch 2.4 with CUDA 12.8
            - MediaPipe, YOLOv11, OpenCV
            - FastAPI, SQLAlchemy, Prometheus
            - 500+ total packages
```

---

## 🎯 Cross-System Development Model

```
┌──────────────────────────────────────────────────────────────┐
│                    MAIN DEVELOPMENT                          │
│                   (RTX 5070 Ti - 12GB)                       │
│  - Active development                                        │
│  - Fast iteration                                            │
│  - All 4 streams pose detection                              │
│  - Memory profiling baseline                                 │
└──────────────────────────────────────────────────────────────┘
                         ↓ pixi sync
                         ↓ rsync projects/
┌──────────────────────────────────────────────────────────────┐
│                 TRAINING MACHINE                             │
│                (RTX 4070 Ti - 12GB Mobile)                   │
│  - Long training runs                                        │
│  - Model optimization                                        │
│  - Lower power consumption                                   │
└──────────────────────────────────────────────────────────────┘
                         ↓ rsync data/
                         ↓ pull results
┌──────────────────────────────────────────────────────────────┐
│                 BACKUP/TESTING                               │
│                (RTX 3070 Ti - 8GB)                           │
│  - Validate cross-system compatibility                       │
│  - 8GB VRAM constraint testing                               │
│  - Performance benchmarking                                  │
└──────────────────────────────────────────────────────────────┘

All systems:
  ✓ Share pixi.lock (single source of truth)
  ✓ CUDA 12.8 configured identically
  ✓ Use make targets for consistency
  ✓ Auto-detect GPU and adjust batch sizes
```

---

## 📊 What Happens When You Run `make install`

```
make install
    ↓
pixi lock --no-environment
    ↓
    Generates pixi.lock with:
    - PyTorch 2.4.* + CUDA 12.8
    - All 500+ dependencies pinned
    - Platform-specific overrides
    ↓
pixi run pip install -e projects/pose_analyzer
    ↓
    Installs pose_analyzer in editable mode
    - src/pose_analyzer/ available as module
    - Changes reflected immediately
    ↓
pixi run pip install -e projects/gpu_optimizer
    ↓
    Installs gpu_optimizer in editable mode
    - src/gpu_optimizer/ available as module
    - Changes reflected immediately
    ↓
✅ Ready to import and use both projects
```

---

## 🔍 Validation Sequence

```
make validate
    ↓
Checks:
  1. PyTorch installed + version check
  2. CUDA available + version 12.8 check
  3. GPU detected + memory check
  4. 11 critical dependencies present
  5. System RAM available check
    ↓
If all pass: ✅ Green light to proceed
If any fails: ❌ See SETUP_COMPLETE_GUIDE.md section 5
```

---

## 📈 Development Progress Tracking

```
Week 1: Foundation
  - [ ] All 3 machines validated
  - [ ] Pixi workspace synced
  - [ ] Models downloaded
  - [ ] GPU baseline profiled

Week 2: Core Features
  - [ ] Pose detection working
  - [ ] 4 streams processing
  - [ ] GPU memory tracked
  - [ ] Unit tests 80%+

Week 3: Biomechanics
  - [ ] Joint angles calculated
  - [ ] Form scoring implemented
  - [ ] Anomaly detection working
  - [ ] Cross-system benchmarks

Week 4: APIs & Monitoring
  - [ ] Both APIs running
  - [ ] WebSocket working
  - [ ] Prometheus metrics exposed
  - [ ] Grafana dashboards

Week 5: Cross-System Testing
  - [ ] All 3 GPUs validated
  - [ ] Performance stable
  - [ ] Multi-machine sync working
  - [ ] No data loss

Week 6: Production
  - [ ] 100% test coverage
  - [ ] Documentation complete
  - [ ] Performance reports
  - [ ] Ready for deployment
```

---

## 💡 Key Decision Points

```
DECISION 1: Pose Detector
├─ MediaPipe (lightweight, easy)
└─ YOLOv11 (production, accurate) ← RECOMMENDED

DECISION 2: Memory Optimization
├─ Gradient Checkpointing (simple)
├─ Tensor Swapping (automatic)
└─ Both (recommended for RTX 3070 Ti with 8GB)

DECISION 3: Monitoring
├─ TensorBoard (simple, local)
├─ Prometheus + Grafana (production)
└─ Both (for complete tracking)

DECISION 4: APIs
├─ FastAPI only
├─ FastAPI + WebSocket
└─ Full setup with both (recommended for real-time)

DECISION 5: Deployment
├─ Local development only
├─ Docker containers
└─ Kubernetes (future)
```

---

## 🚨 Critical Success Factors

```
✅ MUST:
  1. Use CUDA 12.8 consistently (non-negotiable)
  2. Single pixi.lock for all projects
  3. NVIDIA driver 535+ on all systems
  4. Python 3.10+ everywhere
  5. Sync code before switching machines

⚠️ IMPORTANT:
  1. RTX 3070 Ti needs gradient checkpointing for large models
  2. Mobile RTX 4070 Ti has power limits (check BIOS)
  3. Don't mix conda install with pixi (use pixi only)
  4. Keep pixi.lock in version control
  5. Test on all 3 systems before "finishing"

❌ DON'T:
  1. Install PyTorch manually with conda/pip
  2. Use different CUDA versions per machine
  3. Share pixi environments between machines
  4. Ignore VRAM warnings
  5. Skip validation on new machines
```

---

## 🆘 "I'm Stuck" Flowchart

```
Something isn't working
    ↓
1. Did you run `make validate` recently?
    NO → Run it now: make validate
    YES → Continue
    ↓
2. Check error message
    ↓
    Is it about CUDA or GPU?
        YES → See SETUP_COMPLETE_GUIDE.md section 5.1
        NO → Continue
    ↓
    Is it about missing module?
        YES → Run: make install
        NO → Continue
    ↓
    Is it about API/network?
        YES → Check ports: lsof -i :8001
        NO → Continue
    ↓
3. Check logs
    tail -f logs/*.log
    ↓
4. Still stuck?
    Read: PARALLEL_IMPL_PLAN.md Part 9 (Troubleshooting)
    OR
    Run: make clean && make install && make validate
```

---

## 📞 File Reference Guide

| Document | Purpose | Read When |
|----------|---------|-----------|
| IMPLEMENTATION_PACKAGE_SUMMARY.md | What you got | First thing |
| SETUP_COMPLETE_GUIDE.md | Step-by-step setup | Before running anything |
| PARALLEL_IMPL_PLAN.md | Deep architecture | Understanding the system |
| PIXI_ROOT_CONFIG.toml | Pixi configuration | Copying to project |
| POSE_ANALYZER_PYPROJECT.toml | Pose project config | Copying to project |
| GPU_OPTIMIZER_PYPROJECT.toml | GPU project config | Copying to project |
| QUICK_START_SCRIPT.sh | Automated setup | After copying files |
| VALIDATE_ENV_SCRIPT.py | Validation | Troubleshooting |
| MAKEFILE | Common commands | Daily development |
| ROADMAP (this file) | Visual guide | Orientating yourself |

---

**Next Step:** Open `SETUP_COMPLETE_GUIDE.md` section 2 and follow Step 1-5

**Good luck! 🚀**
