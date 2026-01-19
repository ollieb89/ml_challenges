# Day 11: Advanced Llama Refinement Plan

## Goal
Implement three advanced optimization techniques for Llama on 8GB VRAM:
1. **Speculative Decoding**: Llama-8B + Llama-1B draft model
2. **4-bit KV Cache**: QuantizedCache for 16k+ context
3. **SDPA**: Scaled Dot Product Attention for 2x speedup

**Target**: <6.5GB total VRAM with 2 models + quantized cache

## Memory Budget
- Llama-3.1-8B (INT4-NF4): ~5.0GB
- Llama-3.2-1B (INT4-NF4): ~0.7GB
- 4-bit KV Cache (16k): ~0.5GB
- System overhead: ~0.3GB
- **Total**: ~6.5GB ✓

## Implementation Phases

### Phase 1: Setup (30 min)
- Verify: transformers>=4.36, bitsandbytes>=0.41, torch>=2.1
- Download: meta-llama/Llama-3.1-8B, meta-llama/Llama-3.2-1B
- Check SDPA availability

### Phase 2: Quantized KV Cache (1.5 hrs)
- Implement QuantizedCache with 4-bit quantization
- Measure memory: FP16 vs 4-bit (~4x reduction expected)
- Validate quality: perplexity check

### Phase 3: Speculative Decoding (2 hrs)
- Load both models with INT4 (NF4 format)
- Configure assistant_model parameter
- Measure: acceptance rate, speedup (target 1.5-2x)

### Phase 4: Integration & Testing (1.5 hrs)
- Combine all three optimizations
- Profile VRAM at 4k, 8k, 16k context
- Generate report with metrics

## Files to Create
- `scripts/advanced_llama_refinement.py`: Main implementation
- `scripts/benchmark_llama_refinement.py`: Automated testing
- `reports/day11_llama_refinement.md`: Results documentation

## Success Criteria
- ✓ VRAM <6.5GB with both models
- ✓ Support 16k+ context
- ✓ 1.5-2x throughput improvement
- ✓ <5% quality degradation
