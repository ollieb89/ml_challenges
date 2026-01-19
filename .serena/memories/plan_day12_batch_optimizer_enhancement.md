# Day 12 Batch Optimizer Enhancement Plan

## Current State (Baseline)
- **Runtime**: 61s (meets <2min target)
- **Accuracy**: ±2 items
- **Models Tested**: ResNet50 ✓, ViT-B ✓, Llama-7B (stub only)
- **Algorithm**: Binary search
- **Constraints**: VRAM <= 90%, Latency <= 2x baseline

## Limitations Identified
1. **Llama stub**: Uses dummy model, not real LLM
2. **No OOM prediction**: Tries batch sizes until crash
3. **Basic search**: Binary only, could be faster
4. **No caching**: Recalculates every time
5. **Limited metrics**: Missing throughput, cost analysis
6. **No Day 11 integration**: Doesn't use quantized models

## Enhancement Plan

### Phase 1: LLM Integration & Predictive OOM (45min)
**Task 1.1**: Replace Llama stub with real models
- Use unsloth/llama-3-8b-bnb-4bit (INT4)
- Add proper input generation for text models
- Handle both inference and generation workloads

**Task 1.2**: Add pose detection models
- YOLOv11n-pose from pose_analyzer project
- Use actual model from data/models/
- Integrate with video processing workflows

**Task 1.3**: Predictive OOM detection
- Monitor memory allocation patterns
- Predict OOM before attempting batch size
- Use exponential backoff when approaching limit

### Phase 2: Enhanced Search & Caching (30min)
**Task 2.1**: Exponential search + Binary refinement
- Start with exponential jumps (1, 2, 4, 8, 16...)
- Find rough upper bound quickly
- Then binary search for exact optimal

**Task 2.2**: Result caching
- Save results per (GPU model, model name, batch size)
- JSON persistence in reports/batch_optimizer_cache.json
- Auto-load on subsequent runs

**Task 2.3**: Throughput metrics
- Calculate samples/second
- Add cost analysis (time to process N samples)
- Multi-objective scoring

### Phase 3: Advanced Reporting (30min)
**Task 3.1**: Comparison tables
- Before vs After optimization
- Cross-model comparison
- Per-GPU performance matrix

**Task 3.2**: Visualization
- Batch size vs VRAM plot
- Batch size vs throughput plot
- Latency curves

**Task 3.3**: Documentation
- Update ADVANCED_42DAY_CHALLENGE_PLAN.md
- Create reports/day12_batch_optimizer_enhanced.md
- Save benchmark results

## Expected Results
- **LLM support**: Real Llama INT4 models tested ✓
- **Faster search**: 30-40% runtime reduction via exponential search
- **Better metrics**: Throughput, cost, multi-objective scoring
- **Production ready**: Caching, error handling, comprehensive reporting

## Success Criteria
- ✓ Real Llama-8B INT4 tested (not stub)
- ✓ Predictive OOM prevents crashes
- ✓ Results cached and reusable
- ✓ Runtime still <2 minutes
- ✓ Accuracy maintained (±2 items)
