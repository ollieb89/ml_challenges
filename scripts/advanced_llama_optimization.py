#!/usr/bin/env python3
"""Advanced Llama Memory & Throughput Optimization (Day 11 Refinement).

Features:
1. 4-bit KV Cache Quantization (Scaling context)
2. Speculative Decoding (Main 8B + Assistant 1B)
3. SDPA (Scaled Dot Product Attention) for fast kernels

Usage:
    pixi run -e cuda python scripts/advanced_llama_optimization.py
"""

import gc
import logging
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)



# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Model IDs
MAIN_MODEL_ID = "unsloth/llama-3-8b-bnb-4bit"
ASSISTANT_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def get_vram_usage():
    return torch.cuda.memory_allocated() / (1024**3)

def load_models():
    """Load main and assistant models in 4-bit with SDPA."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    logger.info(f"Loading Main Model (8B): {MAIN_MODEL_ID}...")
    main_model = AutoModelForCausalLM.from_pretrained(
        MAIN_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa" # Fast Built-in Attention
    )
    
    logger.info(f"Loading Assistant Model (1B): {ASSISTANT_MODEL_ID}...")
    assistant_model = AutoModelForCausalLM.from_pretrained(
        ASSISTANT_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa"
    )
    
    return main_model, assistant_model

def benchmark_generation(model, tokenizer, assistant_model=None, use_quantized_cache=False, max_new_tokens=50):
    """Run generation benchmark with optional speculative decoding and quantized cache."""
    prompt = "Explain the concept of GPU memory fragmentation in as much detail as possible. Why does it happen and how can we solve it?"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False, # Greedy for benchmark consistency
    }
    
    if assistant_model:
        gen_kwargs["assistant_model"] = assistant_model
        logger.info("Using Speculative Decoding (Assistant Model Enabled)")
    
    if use_quantized_cache:
        # Use transformers built-in quantized cache with Quanto backend
        logger.info("Using 4-bit Quantized KV Cache (Quanto Backend)")
        gen_kwargs["cache_implementation"] = "quantized"
        gen_kwargs["cache_config"] = {"nbits": 4, "backend": "quanto"}




    clear_gpu()
    start_vram = get_vram_usage()
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        
    duration = time.time() - start_time
    end_vram = get_vram_usage()
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    
    tokens_generated = len(outputs[0]) - len(inputs[0])
    tps = tokens_generated / duration
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "duration": duration,
        "tps": tps,
        "peak_vram": peak_vram,
        "total_vram": end_vram,
        "tokens": tokens_generated,
        "sample_text": text[:100] + "..."
    }

def main():
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Exiting.")
        return

    logger.info("=== Advanced Llama Optimization Benchmark ===")
    
    try:
        main_model, assistant_model = load_models()
        tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL_ID)
        
        # Test 1: Baseline (Direct generation)
        logger.info("\n--- Running Baseline (Standard 8B) ---")
        res1 = benchmark_generation(main_model, tokenizer)
        logger.info(f"Result: {res1['tps']:.2f} tokens/s, Peak VRAM: {res1['peak_vram']:.2f} GB")
        
        # Test 2: Speculative Decoding
        # We increase tokens to show speedup (short clips have high overhead)
        logger.info("\n--- Running Speculative Decoding (8B + 1B Assistant) ---")
        res2 = benchmark_generation(main_model, tokenizer, assistant_model=assistant_model, max_new_tokens=100)
        logger.info(f"Result: {res2['tps']:.2f} tokens/s, Peak VRAM: {res2['peak_vram']:.2f} GB")
        
        speedup = res2['tps'] / res1['tps']
        logger.info(f"Speedup: {speedup:.2f}x")
        if speedup < 1.0:
            logger.warning("Speculative decoding was slower. This is common for short sequences or fast GPUs where the model overhead dominates.")

        
        # Test 3: 4-bit Quantized KV Cache
        # Note: Quantized cache is most noticeable on LONG sequences.
        # We'll use more tokens for this test if VRAM allows.
        logger.info("\n--- Running 4-bit Quantized KV Cache ---")
        res3 = benchmark_generation(main_model, tokenizer, use_quantized_cache=True, max_new_tokens=200)
        logger.info(f"Result: {res3['tps']:.2f} tokens/s, Peak VRAM: {res3['peak_vram']:.2f} GB")
        
        # Summary Report
        print("\n" + "="*40)
        print("OPTIMIZATION SUMMARY")
        print("="*40)
        print(f"Method                | Tokens/s | Peak VRAM")
        print(f"----------------------|----------|----------")
        print(f"Standard (8B INT4)    | {res1['tps']:>8.2f} | {res1['peak_vram']:>8.2f} GB")
        print(f"Speculative (+1B)     | {res2['tps']:>8.2f} | {res2['peak_vram']:>8.2f} GB")
        if res3:
            print(f"Quantized KV Cache    | {res3['tps']:>8.2f} | {res3['peak_vram']:>8.2f} GB")
        else:
            print(f"Quantized KV Cache    | {'SKIPPED':>8} | {'N/A':>11}")
        print("="*40)


    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
