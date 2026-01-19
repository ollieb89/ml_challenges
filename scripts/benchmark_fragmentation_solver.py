
import torch
import time
import random
import gc
import logging
from typing import List, Dict
from gpu_optimizer.fragmentation_solver import MemoryFragmentationSolver, SolverConfig

# Disable excessive logging
logging.getLogger("fragmentation_solver").setLevel(logging.WARNING)

def get_stats(device="cuda:0") -> Dict[str, float]:
    torch.cuda.synchronize(device)
    stats = torch.cuda.memory_stats(device)
    res = stats.get("reserved_bytes.all.current", 0)
    alloc = stats.get("allocated_bytes.all.current", 0)
    split = stats.get("inactive_split_bytes.all.current", 0)
    ratio = split / res if res > 0 else 0
    return {
        "reserved_mb": res / 1024**2,
        "allocated_mb": alloc / 1024**2,
        "split_mb": split / 1024**2,
        "ratio": ratio
    }

def high_fragmentation_workload(iterations: int = 100, solver: MemoryFragmentationSolver = None):
    tensors = []
    peak_frag = 0.0
    
    device = torch.device("cuda:0")
    
    for i in range(iterations):
        # Mixed allocations
        for _ in range(50):
            # Many small allocations to stress the allocator
            size = random.randint(1024 * 10, 1024 * 1024 * 2) # 10KB to 2MB
            
            if solver and solver.config.enable_tensor_pool:
                t = solver.tensor_pool.allocate(size, device)
                if t is None:
                    t = torch.empty(size, device=device, dtype=torch.uint8)
            else:
                t = torch.empty(size, device=device, dtype=torch.uint8)
            tensors.append(t)
            
        # Large allocation
        tensors.append(torch.empty(1024 * 1024 * 32, device=device, dtype=torch.uint8)) # 32MB
        
        # Free 70% randomly to induce holes
        random.shuffle(tensors)
        to_free = int(len(tensors) * 0.7)
        for _ in range(to_free):
            tensors.pop()
            
        if solver:
            solver.step()
            peak_frag = max(peak_frag, solver.get_fragmentation_ratio())
        else:
            stats = get_stats()
            peak_frag = max(peak_frag, stats["ratio"])
            
    # Final synchronize to ensure all kernel work is done
    torch.cuda.synchronize(device)
    return peak_frag

def run_benchmark():
    print("Initializing Benchmark...")
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    # Baseline
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    print("\nRunning Baseline (No Solver)...")
    start = time.perf_counter()
    baseline_peak_frag = high_fragmentation_workload(iterations=200)
    torch.cuda.synchronize()
    baseline_time = time.perf_counter() - start
    print(f"Baseline Peak Fragmentation: {baseline_peak_frag:.2%}")
    print(f"Baseline Time: {baseline_time:.2f}s")
    
    # Solver Enabled
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    print("\nRunning with MemoryFragmentationSolver...")
    config = SolverConfig(
        target_frag_ratio=0.10,
        trigger_ratio=0.12,
        hysteresis_steps=3,
        cooldown_steps=10,
        pool_block_size=256 * 1024 * 1024 # 256MB Pool
    )
    solver = MemoryFragmentationSolver(config=config)
    
    start = time.perf_counter()
    solver_peak_frag = high_fragmentation_workload(iterations=200, solver=solver)
    torch.cuda.synchronize()
    solver_time = time.perf_counter() - start
    
    print(f"Solver Peak Fragmentation: {solver_peak_frag:.2%}")
    print(f"Solver Time: {solver_time:.2f}s")
    print(f"Empty Cache Calls: {solver.empty_cache_count}")
    
    # Results
    improvement = (baseline_peak_frag - solver_peak_frag) / baseline_peak_frag if baseline_peak_frag > 0 else 0
    overhead = (solver_time - baseline_time) / baseline_time if baseline_time > 0 else 0
    
    print("\n" + "="*40)
    print("BENCHMARK RESULTS")
    print("="*40)
    print(f"Fragmentation Reduction: {improvement:.2%}")
    print(f"Throughput Overhead: {overhead:.2%}")
    print(f"Baseline Frag: {baseline_peak_frag:.2%}")
    print(f"Solver Frag: {solver_peak_frag:.2%}")
    
    if solver_peak_frag < 0.15:
        print("SUCCESS: Fragmentation stayed below 15%")
    else:
        print("FAILURE: Fragmentation exceeded 15%")
    print("="*40)

if __name__ == "__main__":
    run_benchmark()
