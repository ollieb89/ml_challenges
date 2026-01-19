#!/usr/bin/env python3
"""Llama-7B Memory Optimization Demonstrator (Day 11).

This script demonstrates fitting Llama-7B (actually Llama-3-8B-Instruct)
into 8GB VRAM using:
1. INT8 Quantization (bitsandbytes)
2. INT4 Quantization (bitsandbytes)
3. CPU Offloading (accelerate)

Usage:
    pixi run -e cuda python scripts/optimize_llama.py --mode [baseline|int8|int4|offload]
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add gpu_optimizer to path
project_root = Path(__file__).resolve().parent.parent
gpu_opt_src = project_root / "projects" / "gpu_optimizer" / "src"
if str(gpu_opt_src) not in sys.path:
    sys.path.insert(0, str(gpu_opt_src))

try:
    from gpu_optimizer.memory_tracer import MemoryTracer
except ImportError:
    print("Warning: Could not import MemoryTracer. Proceeding without granular tracing.")
    MemoryTracer = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Model ID (Use cached Unsloth model which is already 4-bit optimized)
MODEL_ID = "unsloth/llama-3-8b-bnb-4bit"

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def get_vram_usage():
    return torch.cuda.memory_allocated() / (1024**3)

def run_inference(model, tokenizer, device="cuda"):
    """Run a quick generation to ensure model is working."""
    input_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    logger.info("Running inference...")
    start_time = time.time()
    
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=20)
        
    duration = time.time() - start_time
    logger.info(f"Inference complete in {duration:.2f}s")

def baseline_fp16():
    """Attempt to load in FP16 (Will likely OOM on 8GB)."""
    logger.info(f"Attempting Baseline FP16 load of {MODEL_ID}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto" # Let accelerate figure it out, but it might just OOM if no offload
        )
        return model
    except Exception as e:
        logger.error(f"Baseline FP16 failed: {e}")
        raise

def int8_quantization():
    """Load in INT8."""
    logger.info(f"Loading {MODEL_ID} in INT8...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    return model

def int4_quantization():
    """Load in INT4."""
    logger.info(f"Loading {MODEL_ID} in INT4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", # Normal Float 4
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    return model

def cpu_offload_fp16():
    """Load FP16 with CPU offloading."""
    logger.info(f"Loading {MODEL_ID} in FP16 with CPU Offloading...")
    # device_map="auto" with offload_folder enables offloading
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload",
        offload_state_dict=True
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="Llama Memory Optimization")
    parser.add_argument("--mode", choices=["baseline", "int8", "int4", "offload"], required=True)
    args = parser.parse_args()
    
    clear_gpu()
    
    tracer = None
    if MemoryTracer:
        tracer = MemoryTracer(max_events=5000)
    
    try:
        logger.info(f"Initial VRAM: {get_vram_usage():.2f} GB")
        
        if tracer:
            pass # Trace only inference for now to avoid complexity with model hooks during loading
            
        model = None
        if args.mode == "baseline":
            model = baseline_fp16()
        elif args.mode == "int8":
            model = int8_quantization()
        elif args.mode == "int4":
            model = int4_quantization()
        elif args.mode == "offload":
            model = cpu_offload_fp16()
            
        logger.info(f"Model loaded. VRAM: {get_vram_usage():.2f} GB")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # Trace Inference
        if tracer:
            with tracer.trace(model):
                run_inference(model, tokenizer)
        else:
            run_inference(model, tokenizer)
            
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        logger.info(f"Peak VRAM during inference: {peak_vram:.2f} GB")
        
        if tracer:
            tracer.print_summary()
            
    except torch.cuda.OutOfMemoryError:
        logger.error("❌ OOM: GPU ran out of memory!")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    finally:
        if tracer and tracer.is_tracing:
            tracer.stop_tracing()

if __name__ == "__main__":
    main()
