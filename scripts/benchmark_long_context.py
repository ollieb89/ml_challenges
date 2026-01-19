#!/usr/bin/env python3
"""Benchmark for Llama KV Cache Optimization (Day 11 Refinement).

Demonstrates the VRAM savings of 4-bit KV Cache quantization across different context lengths.

Usage:
    pixi run -e cuda python scripts/benchmark_long_context.py
"""

import gc
import logging
import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Model ID
MODEL_ID = "unsloth/llama-3-8b-bnb-4bit"

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def get_vram_usage():
    return torch.cuda.memory_allocated() / (1024**3)

def run_context_test(model, tokenizer, context_length, use_quantized_cache=False):
    """Measure VRAM for a specific context length."""
    clear_gpu()
    
    # Create long prompt by repeating a string
    base_text = "The quick brown fox jumps over the lazy dog. "
    repeats = (context_length // 10) + 1
    input_text = (base_text * repeats)[:context_length]
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=context_length).to("cuda")
    actual_len = inputs.input_ids.shape[1]
    
    gen_kwargs = {
        "max_new_tokens": 1, # Just one token to measure cache impact
        "do_sample": False,
    }
    
    if use_quantized_cache:
        # 4-bit quantization with axis 0
        logger.info("Using 4-bit Quantized KV Cache (Dictionary Config)")
        gen_kwargs["cache_implementation"] = "quantized"
        gen_kwargs["cache_config"] = {"nbits": 4, "backend": "quanto", "axis_key": 0}


    logger.info(f"Testing length={actual_len}, quantized={use_quantized_cache}...")
    
    start_vram = get_vram_usage()
    
    try:
        with torch.no_grad():
            _ = model.generate(**inputs, **gen_kwargs)
            
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        cache_vram = peak_vram - start_vram
        
        return peak_vram, cache_vram
    except torch.cuda.OutOfMemoryError:
        logger.error(f"OOM at length {actual_len} (quantized={use_quantized_cache})")
        return None, None

def main():
    if not torch.cuda.is_available():
        return

    # Load model once
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    logger.info(f"Loading {MODEL_ID} for KV Cache Benchmark...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    context_lengths = [1024, 2048, 4096, 6144, 8192]
    standard_vram = []
    quantized_vram = []
    
    print("\n--- KV Cache VRAM Benchmark ---")
    print(f"{'Length':<10} | {'FP16 Cache (GB)':<15} | {'4-bit Cache (GB)':<15} | {'Savings':<10}")
    print("-" * 60)
    
    for length in context_lengths:
        # Standard test
        peak_std, cache_std = run_context_test(model, tokenizer, length, use_quantized_cache=False)
        # Quantized test
        peak_quant, cache_quant = run_context_test(model, tokenizer, length, use_quantized_cache=True)
        
        std_str = f"{cache_std:.2f}" if cache_std else "OOM"
        quant_str = f"{cache_quant:.2f}" if cache_quant else "OOM"
        savings = f"{(1 - cache_quant/cache_std)*100:.1f}%" if (cache_std and cache_quant) else "N/A"
        
        print(f"{length:<10} | {std_str:<15} | {quant_str:<15} | {savings:<10}")
        
        standard_vram.append(cache_std if cache_std else 0)
        quantized_vram.append(cache_quant if cache_quant else 0)

    # Plotting
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(context_lengths, standard_vram, 'o-', label='Standard (FP16) Cache')
        plt.plot(context_lengths, quantized_vram, 's-', label='Quantized (4-bit) Cache')
        plt.xlabel('Context Length (tokens)')
        plt.ylabel('Cache Memory (GB)')
        plt.title('Llama-3-8B KV Cache Memory: FP16 vs 4-bit')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        output_path = "reports/kv_cache_scaling.png"
        Path("reports").mkdir(exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Benchmark plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}")

if __name__ == "__main__":
    main()
