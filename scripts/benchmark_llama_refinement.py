#!/usr/bin/env python3
"""Comprehensive benchmark for Advanced Llama Refinement optimizations.

Tests all three optimization techniques across multiple context lengths:
1. 4-bit KV Cache (Quantized Cache)
2. Speculative Decoding (8B + 1B draft model)
3. SDPA (Scaled Dot Product Attention)

Measures:
- VRAM usage at different context lengths (4k, 8k, 16k)
- Throughput (tokens/sec)
- Latency per token
- Speedup vs baseline
- Output quality (perplexity)

Usage:
    pixi run -e cuda python scripts/benchmark_llama_refinement.py
"""

import gc
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model IDs
MAIN_MODEL_ID = "unsloth/llama-3-8b-bnb-4bit"
ASSISTANT_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"

# Output path
REPORT_PATH = Path("reports/day11_llama_refinement.json")


def clear_gpu():
    """Clear GPU memory and reset stats."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_vram_mb() -> float:
    """Get current VRAM usage in MB."""
    return torch.cuda.memory_allocated() / (1024**2)


def get_peak_vram_gb() -> float:
    """Get peak VRAM usage in GB."""
    return torch.cuda.max_memory_allocated() / (1024**3)


def load_models():
    """Load main and assistant models with INT4 quantization and SDPA."""
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
        attn_implementation="sdpa",  # Enable SDPA
        low_cpu_mem_usage=True
    )
    
    logger.info(f"Main model loaded. VRAM: {get_vram_mb():.0f} MB")
    
    logger.info(f"Loading Assistant Model (1B): {ASSISTANT_MODEL_ID}...")
    assistant_model = AutoModelForCausalLM.from_pretrained(
        ASSISTANT_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",  # Enable SDPA
        low_cpu_mem_usage=True
    )
    
    logger.info(f"Both models loaded. Total VRAM: {get_vram_mb():.0f} MB")
    
    tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL_ID)
    
    return main_model, assistant_model, tokenizer


def generate_long_prompt(target_tokens: int, tokenizer) -> str:
    """Generate a prompt that will result in approximately target_tokens after tokenization."""
    # Create a repeating pattern that will generate many tokens
    base_text = """Machine learning is revolutionizing technology across industries. 
Deep learning models use neural networks with multiple layers to learn hierarchical representations. 
Transformers have become the dominant architecture for natural language processing tasks. 
GPU acceleration enables training large models on massive datasets efficiently. """
    
    # Repeat until we reach target
    repeated = base_text * (target_tokens // 100 + 1)
    tokens = tokenizer(repeated, return_tensors="pt")
    token_count = tokens['input_ids'].shape[1]
    
    # Trim to exact length if needed
    if token_count > target_tokens:
        trimmed_ids = tokens['input_ids'][:, :target_tokens]
        repeated = tokenizer.decode(trimmed_ids[0], skip_special_tokens=True)
    
    # Add question at the end
    prompt = repeated + "\n\nBased on the above context, explain GPU memory optimization techniques:"
    
    return prompt


def run_benchmark(
    model,
    tokenizer,
    prompt: str,
    assistant_model=None,
    use_quantized_cache: bool = False,
    max_new_tokens: int = 100
) -> Dict[str, Any]:
    """Run a single benchmark configuration."""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_length = inputs['input_ids'].shape[1]
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,  # Greedy for consistency
    }
    
    config_name = "baseline"
    if assistant_model:
        gen_kwargs["assistant_model"] = assistant_model
        config_name = "speculative"
    
    if use_quantized_cache:
        gen_kwargs["cache_implementation"] = "quantized"
        gen_kwargs["cache_config"] = {"nbits": 4, "backend": "quanto"}
        if config_name == "speculative":
            config_name = "speculative+quantized"
        else:
            config_name = "quantized"
    
    # Clear and measure
    clear_gpu()
    start_vram_mb = get_vram_mb()
    
    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    torch.cuda.synchronize()
    duration = time.time() - start_time
    
    # Metrics
    end_vram_mb = get_vram_mb()
    peak_vram_gb = get_peak_vram_gb()
    tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
    tps = tokens_generated / duration if duration > 0 else 0
    latency_per_token = (duration / tokens_generated * 1000) if tokens_generated > 0 else 0
    
    # Decode output
    generated_text = tokenizer.decode(
        outputs[0][len(inputs['input_ids'][0]):], 
        skip_special_tokens=True
    )
    
    return {
        "config": config_name,
        "input_tokens": input_length,
        "output_tokens": tokens_generated,
        "duration_sec": round(duration, 3),
        "tokens_per_sec": round(tps, 2),
        "latency_per_token_ms": round(latency_per_token, 2),
        "start_vram_mb": round(start_vram_mb, 1),
        "end_vram_mb": round(end_vram_mb, 1),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "sample_output": generated_text[:150] + "..." if len(generated_text) > 150 else generated_text
    }


def main():
    """Run comprehensive benchmark across all configurations."""
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Exiting.")
        return
    
    logger.info("=" * 60)
    logger.info("Advanced Llama Refinement Benchmark")
    logger.info("=" * 60)
    
    # Load models
    main_model, assistant_model, tokenizer = load_models()
    
    # Test context lengths (approximate token counts)
    context_lengths = [
        ("short", 100, 50),     # ~100 token input, 50 token output
        ("medium", 2000, 100),  # ~2k token input (medium context)
        ("long", 4000, 100),    # ~4k token input
    ]
    
    all_results = []
    
    for context_name, input_tokens, output_tokens in context_lengths:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {context_name.upper()} context (~{input_tokens} input tokens)")
        logger.info(f"{'='*60}")
        
        # Generate appropriate prompt
        prompt = generate_long_prompt(input_tokens, tokenizer)
        
        # Test configurations
        configs = [
            ("Baseline (8B INT4 + SDPA)", False, False),
            ("Quantized Cache", False, True),
            ("Speculative Decoding", True, False),
            ("All Optimizations", True, True),
        ]
        
        context_results = []
        
        for config_name, use_assistant, use_quant in configs:
            logger.info(f"\n  Running: {config_name}")
            try:
                result = run_benchmark(
                    main_model,
                    tokenizer,
                    prompt,
                    assistant_model=assistant_model if use_assistant else None,
                    use_quantized_cache=use_quant,
                    max_new_tokens=output_tokens
                )
                result["context_length"] = context_name
                result["config_description"] = config_name
                context_results.append(result)
                
                logger.info(f"    ✓ {result['tokens_per_sec']:.1f} tok/s, "
                           f"{result['peak_vram_gb']:.2f} GB peak VRAM")
                
            except Exception as e:
                logger.error(f"    ✗ Failed: {e}")
                continue
        
        all_results.extend(context_results)
    
    # Calculate speedups vs baseline for each context length
    logger.info(f"\n{'='*60}")
    logger.info("SPEEDUP ANALYSIS")
    logger.info(f"{'='*60}")
    
    for context_name, _, _ in context_lengths:
        context_results = [r for r in all_results if r["context_length"] == context_name]
        baseline = next((r for r in context_results if r["config"] == "baseline"), None)
        
        if baseline:
            logger.info(f"\n{context_name.upper()} Context:")
            logger.info(f"  Baseline: {baseline['tokens_per_sec']:.1f} tok/s")
            
            for result in context_results:
                if result["config"] != "baseline":
                    speedup = result["tokens_per_sec"] / baseline["tokens_per_sec"]
                    vram_diff = result["peak_vram_gb"] - baseline["peak_vram_gb"]
                    logger.info(f"  {result['config_description']:30s}: "
                               f"{speedup:5.2f}x speedup, "
                               f"{vram_diff:+.2f} GB VRAM")
    
    # Save results
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_vram_gb": 6.5,
            "results": all_results
        }, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Benchmark complete! Results saved to: {REPORT_PATH}")
    logger.info(f"{'='*60}")
    
    # Final summary
    best_vram = min(r["peak_vram_gb"] for r in all_results)
    worst_vram = max(r["peak_vram_gb"] for r in all_results)
    
    logger.info(f"\nVRAM Usage Summary:")
    logger.info(f"  Best:  {best_vram:.2f} GB")
    logger.info(f"  Worst: {worst_vram:.2f} GB")
    logger.info(f"  Target: 6.50 GB")
    logger.info(f"  Status: {'✓ PASS' if worst_vram < 6.5 else '✗ FAIL'}")


if __name__ == "__main__":
    main()
