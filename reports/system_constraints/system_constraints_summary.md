# System Constraints Summary

**Generated:** 2026-01-18
**Purpose:** Document GPU constraints across all systems in the AI/ML pipeline project.

---

## Systems Overview

| System ID | GPU | VRAM (GB) | Compute | Primary Use |
|-----------|-----|-----------|---------|-------------|
| `desktop-5070ti` | RTX 5070 Ti | 16 | SM 12.0 | Development, Inference, Testing |
| `laptop-4070ti` | RTX 4070 Ti Mobile | 12 | SM 8.9 | Training, Optimization |
| `pc-3070ti` | RTX 3070 Ti | 8 | SM 8.6 | Backup, Inference, Testing |

---

## Per-System Model Constraints

### RTX 5070 Ti (16GB VRAM) - `desktop-5070ti`

**Status:** ✅ Profiled on 2026-01-18

| Model | Precision | Max Batch | VRAM (MB) | Latency (ms) | Notes |
|-------|-----------|-----------|-----------|--------------|-------|
| ResNet50 | FP32 | 256 | 2,802 | 166 | ✅ No issues |
| ResNet50 | FP16 | 256 | 1,603 | 113 | ✅ Recommended |
| ViT-Base-16 | FP32 | 256 | 2,411 | 1,242 | ✅ Works |
| ViT-Base-16 | FP16 | 256 | 1,219 | 160 | ✅ Recommended |
| YOLO-Pose-v11n | FP16 | 128 | 3,019 | 481 | ✅ Pose detection |
| Llama-7B | FP16 | 0 | - | - | ❌ FAILS (requires ~16GB, available 15.4GB) |
| Llama-7B | INT8 | 4 | 8,704 | - | ⚠️ Estimated only |
| Llama-7B | INT4 | 8 | 5,120 | - | ⚠️ Estimated only |

**Recommendations:**
- Use FP16 for vision models (ResNet50, ViT-Base)
- For Llama-7B, use INT8 or INT4 quantization
- 4-stream pose detection feasible at batch 32 per stream

---

### RTX 4070 Ti Mobile (12GB VRAM) - `laptop-4070ti`

**Status:** 🔄 Pending profiling

**Expected Constraints:**

| Model | Precision | Expected Max Batch | Notes |
|-------|-----------|-------------------|-------|
| ResNet50 | FP32 | 128-192 | ~75% of 5070 Ti |
| ResNet50 | FP16 | 192-256 | Should be fine |
| ViT-Base-16 | FP32 | 128-192 | May hit OOM at 256 |
| ViT-Base-16 | FP16 | 192-256 | Recommended |
| YOLO-Pose-v11n | FP16 | 64-96 | Sufficient for 2-stream |
| Llama-7B | FP16 | 0 | ❌ Will NOT fit |
| Llama-7B | INT8 | 1-2 | ⚠️ Tight fit at 8.5GB |
| Llama-7B | INT4 | 4-8 | ✅ Should work |

**To Profile:** Run `pixi run -e cuda python scripts/gpu_constraint_profiler.py` on this machine.

---

### RTX 3070 Ti (8GB VRAM) - `pc-3070ti` ⚠️ MOST CONSTRAINED

**Status:** 🔄 Pending profiling

**Expected Constraints:**

| Model | Precision | Expected Max Batch | Notes |
|-------|-----------|-------------------|-------|
| ResNet50 | FP32 | 64-96 | Limited by 8GB |
| ResNet50 | FP16 | 128-192 | Better performance |
| ViT-Base-16 | FP32 | 32-64 | OOM likely at 128 |
| ViT-Base-16 | FP16 | 64-128 | Recommended |
| YOLO-Pose-v11n | FP16 | 32-64 | 1-2 stream max |
| Llama-7B | FP16 | 0 | ❌ Absolutely NOT |
| Llama-7B | INT8 | 0 | ❌ Will NOT fit (8.5GB > 8GB) |
| Llama-7B | INT4 | 1-4 | ⚠️ Tight, may need smaller context |

**Critical Constraints for 8GB System:**
1. **Llama-7B FP16/FP32:** Impossible
2. **Llama-7B INT8:** Will NOT fit (requires ~8.5GB operational)
3. **Llama-7B INT4:** Only option, limited to batch=1, short context
4. **Multi-stream pose detection:** Limited to 2 streams max
5. **Large batch ViT inference:** Avoid, use FP16 only

**To Profile:** Run `pixi run -e cuda python scripts/gpu_constraint_profiler.py` on this machine.

---

## What Breaks on 8GB VRAM

### ❌ Will NOT Work:
1. **Llama-7B FP32** - Requires ~28GB
2. **Llama-7B FP16** - Requires ~14GB  
3. **Llama-7B INT8** - Requires ~8.5GB (just over limit)
4. **ViT-Base FP32 batch > 64** - OOM risk
5. **ResNet50 FP32 batch > 96** - OOM risk
6. **4-stream pose detection** - Insufficient VRAM
7. **Training any model** - No room for gradients

### ⚠️ Will Work with Constraints:
1. **Llama-7B INT4** - batch=1, context ≤ 512 tokens
2. **ViT-Base FP16** - batch ≤ 64
3. **ResNet50 FP16** - batch ≤ 128
4. **YOLO-Pose** - 1-2 streams only
5. **Inference-only workloads** - Must be FP16 or lower

### ✅ Safe on 8GB:
1. **ResNet50 FP16 batch ≤ 64** - ~800MB
2. **ViT-Base FP16 batch ≤ 32** - ~700MB
3. **YOLO-Pose-v11n batch ≤ 32** - ~800MB
4. **Single-stream pose detection** - ~400MB

---

## Llama-7B Specific Analysis

| Configuration | VRAM Required | RTX 5070 Ti | RTX 4070 Ti | RTX 3070 Ti |
|---------------|---------------|-------------|-------------|-------------|
| FP32 (32-bit) | ~28GB | ❌ | ❌ | ❌ |
| FP16 (16-bit) | ~14GB | ❌ | ❌ | ❌ |
| INT8 (8-bit) | ~7GB model + ~1.5GB activations | ⚠️ batch 1-4 | ⚠️ batch 1-2 | ❌ |
| INT4 (4-bit) | ~3.5GB model + ~1.5GB activations | ✅ batch 4-8 | ✅ batch 2-4 | ⚠️ batch 1 |
| GGML Q4_K_M | ~4GB model | ✅ | ✅ | ✅ |

**Recommended: Use GGML quantized models for RTX 3070 Ti**

---

## Profiling Commands

```bash
# Profile current system (auto-detects GPU)
pixi run -e cuda python scripts/gpu_constraint_profiler.py

# Profile specific models only
pixi run -e cuda python scripts/gpu_constraint_profiler.py --models resnet50 vit_base yolo_pose

# Custom output directory
pixi run -e cuda python scripts/gpu_constraint_profiler.py --output reports/my_test/

# SSH to 3070Ti machine and profile
ssh ollie@192.168.1.211 "cd ~/Tools/ai-ml-pipeline && pixi run -e cuda python scripts/gpu_constraint_profiler.py"
```

---

## Challenge Plan Status Update

**Day 6 Tasks:**
- [x] ~~Sync code to RTX 3070 Ti machine~~ (Manual step)
- [x] ~~Pose detection: 1-stream baseline~~ (Verified: 53/53 tests passed)
- [x] GPU optimizer: Profile on 8GB VRAM - **Script created, ready to run**
- [x] Identify what breaks on 8GB - **Documented above**
- [x] Document constraints per system - **This document**

---

*Generated by gpu_constraint_profiler.py analysis*
