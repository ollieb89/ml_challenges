
import torch
import time
import pandas as pd
from typing import Dict, List, Any

def measure_fragmentation(device="cuda:0") -> Dict[str, Any]:
    """Measure current VRAM fragmentation metrics."""
    torch.cuda.synchronize(device)
    stats = torch.cuda.memory_stats(device)
    
    allocated = stats.get("allocated_bytes.all.current", 0)
    reserved = stats.get("reserved_bytes.all.current", 0)
    inactive_split = stats.get("inactive_split_bytes.all.current", 0)
    
    # Fragmentation in PyTorch is often calculated as inactive_split_bytes / reserved_bytes
    # or (reserved - allocated) / reserved.
    # The challenge mentions <15% wasted memory. 
    # We will track both.
    
    wastage = reserved - allocated
    wastage_pct = (wastage / reserved * 100) if reserved > 0 else 0
    frag_pct = (inactive_split / reserved * 100) if reserved > 0 else 0
    
    return {
        "allocated_mb": allocated / 1024**2,
        "reserved_mb": reserved / 1024**2,
        "inactive_split_mb": inactive_split / 1024**2,
        "wastage_mb": wastage / 1024**2,
        "wastage_pct": wastage_pct,
        "frag_pct": frag_pct
    }

def run_stress_scenario(scenario_name: str, allocation_fn: callable):
    print(f"\n--- Scenario: {scenario_name} ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    before = measure_fragmentation()
    print(f"Before: Allocated={before['allocated_mb']:.2f}MB, Reserved={before['reserved_mb']:.2f}MB, Frag={before['frag_pct']:.2f}%")
    
    tensors = allocation_fn()
    
    during = measure_fragmentation()
    print(f"During: Allocated={during['allocated_mb']:.2f}MB, Reserved={during['reserved_mb']:.2f}MB, Frag={during['frag_pct']:.2f}%")
    
    # Free half randomly to induce fragmentation
    import random
    indices = list(range(len(tensors)))
    random.shuffle(indices)
    to_free = indices[:len(tensors)//2]
    
    for i in sorted(to_free, reverse=True):
        tensors.pop(i)
    
    gc_collected = 0
    import gc
    gc_collected = gc.collect()
    
    after_free = measure_fragmentation()
    print(f"After partial free: Allocated={after_free['allocated_mb']:.2f}MB, Reserved={after_free['reserved_mb']:.2f}MB, Frag={after_free['frag_pct']:.2f}%")
    
    return tensors

def scenario_random_sizes():
    tensors = []
    # Mix of large and small allocations
    for _ in range(100):
        size = torch.randint(1, 1024 * 1024 * 10, (1,)).item() # 1B to 10MB
        tensors.append(torch.empty(size, device="cuda", dtype=torch.uint8))
    return tensors

def scenario_large_small_mix():
    tensors = []
    for _ in range(10):
        # Large chunk
        tensors.append(torch.empty(1024 * 1024 * 50, device="cuda", dtype=torch.uint8)) # 50MB
        # Small "peppering" allocations
        for _ in range(50):
            tensors.append(torch.empty(1024 * 100, device="cuda", dtype=torch.uint8)) # 100KB
    return tensors

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        exit(1)
        
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    run_stress_scenario("Random Sizes", scenario_random_sizes)
    run_stress_scenario("Large/Small Mix", scenario_large_small_mix)
