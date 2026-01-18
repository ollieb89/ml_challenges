# GPU Constraint Profiling - Checkpoint

**Date:** 2026-01-18
**Status:** Day 6 Tasks Complete

## Completed Work

### 1. Created GPU Constraint Profiler Script
- **File:** `scripts/gpu_constraint_profiler.py`
- **Features:**
  - Auto-detects GPU and VRAM
  - Profiles ResNet50, ViT-Base (FP32/FP16)
  - Profiles YOLO-Pose-v11n
  - Estimates Llama-7B requirements (FP16/INT8/INT4)
  - Tests batch sizes until OOM
  - Outputs JSON + Markdown reports

### 2. Validated on RTX 5070 Ti
- **Report:** `reports/system_constraints/nvidia_geforce_rtx_5070_ti_15gb_constraints.json`
- **Key findings:**
  - ResNet50 FP32 batch 256: 2.8GB, 166ms
  - ViT-Base FP16 batch 256: 1.2GB, 160ms
  - YOLO-Pose batch 128: 3GB, 481ms
  - Llama-7B FP16: FAILS (requires 16GB, only 15.4GB available)

### 3. Documented 8GB Constraints
- **Doc:** `reports/system_constraints/system_constraints_summary.md`
- **What breaks on 8GB:**
  - Llama-7B FP32/FP16/INT8 all FAIL
  - Llama-7B INT4 is ONLY option
  - Multi-stream pose limited to 2 streams

### 4. Updated Challenge Plan
- Marked Day 6 GPU optimizer tasks complete
- Added detailed notes on what breaks

## Next Steps
1. Run profiler on actual RTX 3070 Ti machine via SSH
2. SSH command: `ssh ollie@192.168.1.211 "cd ~/Tools/ai-ml-pipeline && pixi run -e cuda python scripts/gpu_constraint_profiler.py"`
3. Proceed to Day 7: 4-Stream Concurrent Pose Detection

## Files Created/Modified
- `scripts/gpu_constraint_profiler.py` (NEW)
- `reports/system_constraints/nvidia_geforce_rtx_5070_ti_15gb_constraints.json` (NEW)
- `reports/system_constraints/nvidia_geforce_rtx_5070_ti_15gb_constraints.md` (NEW)
- `reports/system_constraints/system_constraints_summary.md` (NEW)
- `docs/ADVANCED_42DAY_CHALLENGE_PLAN.md` (UPDATED)
