# Advanced Llama Optimization Patterns (8GB VRAM Constraint)

This memory document outlines advanced patterns for fitting large language models (LLMs) into constrained VRAM (specifically 8GB) while maintaining high throughput and handling longer contexts.

## 1. Multi-Tier Quantization
- **Weights (INT4-NF4)**: Reduces Llama-3-8B from 16GB to ~5.3GB. NF4 (NormalFloat4) is preferred over standard INT4 for maintaining dynamic range.
- **KV Cache (4-bit)**: Standard FP16 KV cache consumes ~128KB per token for Llama-3-8B. 4-bit quantization reduces this to ~32KB per token. 
  - *Impact*: Allows 32k context in ~1GB VRAM, whereas FP16 would require 4GB.

## 2. Speculative Decoding (Draft-and-Verify)
- **Concept**: Use a significantly smaller model (e.g., Llama-3.2-1B) to draft multiple candidate tokens in a single forward pass, then use the 8B model to verify them in one pass.
- **Memory Overhead**: 1B model (INT4) adds ~700-800MB.
- **Combined Footprint**: 8B (5.3GB) + 1B (0.8GB) + KV Cache (1GB) + System Overhead = ~7.5GB.
- **Performance**: Can achieve 1.5x - 2.5x speedup in throughput and latency.

## 3. Attention Kernels (SDPA & FlashAttention-2)
- **SDPA (Scaled Dot Product Attention)**: Default in PyTorch 2.0+, provides significant speedup and memory savings over manual attention loops.
- **FlashAttention-2**: Even more memory-efficient, particularly for long sequences.

## 4. Context Eviction (StreamingLLM)
- For infinite or extremely long sequences, we implement a rolling window that preserves the "Attention Sinks" (first 4 tokens) and the most recent N tokens.
- This prevents OOM regardless of sequence length, with minimal impact on coherence for most tasks.

## 5. Implementation Strategy
- Use `transformers.cache_utils.QuantizedCache` for context management.
- Use `assistant_model` parameter in `model.generate()` for native speculative decoding.
- Use `bitsandbytes` with `bnb_4bit_compute_dtype=torch.float16` for optimal speed on consumer GPUs.
