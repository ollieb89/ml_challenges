import torch
import torchvision
import time
import argparse
import sys
import os

# Add src to path just in case, though editable install should handle it
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from gpu_optimizer.tensor_swapper import TensorSwapper
import torch.optim as optim
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Benchmark")

def train_step(model, inputs, targets, criterion, optimizer, swapper=None):
    optimizer.zero_grad()
    
    if swapper:
        with swapper.enable():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
    else:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
    optimizer.step()
    return loss.item()

def run_benchmark(batch_size=32, steps=10, swap=False, threshold=0.8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA not available. Benchmark invalid for GPU scraping.")
        # Continue anyway for logic test if needed, but return dummy values
        # return 0, 0
    
    logger.info(f"Setting up ResNet50 | BS={batch_size} | Swap={swap} | Threshold={threshold}")
    
    try:
        model = torchvision.models.resnet50()
        model.to(device)
        model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        inputs = torch.randn(batch_size, 3, 224, 224, device=device)
        targets = torch.randint(0, 1000, (batch_size,), device=device)
        
        swapper = TensorSwapper(threshold_percent=threshold, device=device) if swap else None
        
        # Warmup
        logger.info("Warming up...")
        for _ in range(3):
            train_step(model, inputs, targets, criterion, optimizer, swapper)
            
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if swapper: swapper.reset_stats()
        
        start_time = time.time()
        
        logger.info(f"Running {steps} steps...")
        for i in range(steps):
            train_step(model, inputs, targets, criterion, optimizer, swapper)
            
        torch.cuda.synchronize()
        duration = time.time() - start_time
        
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        avg_latency = (duration / steps) * 1000
        
        logger.info(f"Results for Swap={swap}:")
        logger.info(f"  Avg Latency: {avg_latency:.2f} ms")
        logger.info(f"  Peak Memory: {peak_mem:.2f} GB")
        if swapper:
            logger.info(f"  Stats: {swapper.get_stats()}")
        
        return avg_latency, peak_mem

    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error(f"OOM with Batch Size {batch_size}")
            torch.cuda.empty_cache()
            return None, None
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    
    print(f"\n{'='*20} RUNNING BENCHMARK {'='*20}")
    
    # Baseline
    lat_base, mem_base = run_benchmark(batch_size=args.batch_size, steps=args.steps, swap=False)
    
    if lat_base is None:
        print("Baseline OOM. Proceeding to test Swapped configuration to verify feasibility...")
    
    # Swapping (Force Swap to measure worst-case overhead)
    # Threshold 0.0 forces swapping of everything qualified
    lat_swap, mem_swap = run_benchmark(batch_size=args.batch_size, steps=args.steps, swap=True, threshold=0.0)
    
    if lat_swap:
        if lat_base:
            overhead = ((lat_swap - lat_base) / lat_base) * 100
            mem_saved = ((mem_base - mem_swap) / mem_base) * 100
            
            print("\n" + "="*40)
            print("BENCHMARK SUMMARY")
            print("="*40)
            print(f"Batch Size: {args.batch_size}")
            print(f"Baseline: {lat_base:.2f} ms | {mem_base:.2f} GB")
            print(f"Swapped:  {lat_swap:.2f} ms | {mem_swap:.2f} GB")
            print("-" * 40)
            print(f"Latency Impact: {overhead:+.2f}%")
            print(f"Memory Savings: {mem_saved:+.2f}%")
            print("="*40)
        else:
            print("\n" + "="*40)
            print("BENCHMARK SUMMARY (Baseline Failed/OOM)")
            print("="*40)
            print(f"Batch Size: {args.batch_size}")
            print(f"Baseline: OOM")
            print(f"Swapped:  {lat_swap:.2f} ms | {mem_swap:.2f} GB")
            print("-" * 40)
            print("Swapping enabled successful execution where Baseline failed.")
            print("="*40)
    else:
        print("Swapped run failed (OOM?)")
