
import torch
import time
import logging
import gc
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

from shared_utils.metrics import metrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fragmentation_solver")

@dataclass
class SolverConfig:
    """Configuration for the Memory Fragmentation Solver."""
    target_frag_ratio: float = 0.15
    trigger_ratio: float = 0.20
    cooldown_steps: int = 10
    max_empty_cache_calls_per_epoch: int = 5
    hysteresis_steps: int = 3
    enable_tensor_pool: bool = True
    pool_block_size: int = 1024 * 1024 * 64  # 64MB default block size for the pool

class TensorPool:
    """
    Sub-allocator for small tensors to prevent fragmentation.
    Groups multiple small allocations into a single large contiguous block.
    """
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.pool: Optional[torch.Tensor] = None
        self.offset = 0
        self.active_allocations = 0

    def allocate(self, size_bytes: int, device: torch.device) -> Optional[torch.Tensor]:
        # Only handle allocations smaller than 10% of block size to preserve coalescing logic
        if size_bytes > self.block_size // 10:
            return None
        
        if self.pool is None or (self.offset + size_bytes > self.block_size):
            if self.active_allocations > 0:
                # We have a leak or long-lived tensor if we need a new block while the old is busy
                # In a real implementation, we'd manage multiple blocks.
                logger.debug("Pool block full, active allocations still present.")
            
            try:
                self.pool = torch.empty(self.block_size, device=device, dtype=torch.uint8)
                self.offset = 0
                self.active_allocations = 0
            except torch.cuda.OutOfMemoryError:
                return None

        # Return a view/slice of the pool
        tensor_view = self.pool[self.offset : self.offset + size_bytes]
        self.offset += size_bytes
        self.active_allocations += 1
        return tensor_view

    def reset(self):
        """Reset the pool. Call this when it's safe (e.g., between iterations)."""
        self.pool = None
        self.offset = 0
        self.active_allocations = 0

class MemoryFragmentationSolver:
    """
    Solver for reducing VRAM fragmentation in PyTorch.
    
    Monitors inactive_split_bytes and uses hysteresis to trigger safe 
    cache clearing and preemptive allocation strategies.
    """
    def __init__(self, config: Optional[SolverConfig] = None, device: str = "cuda:0"):
        self.config = config or SolverConfig()
        self.device = torch.device(device)
        self.tensor_pool = TensorPool(self.config.pool_block_size)
        
        # State tracking
        self.high_frag_counter = 0
        self.cooldown_counter = 0
        self.empty_cache_count = 0
        self.steps_since_last_check = 0
        
        # Stats logging
        self.history: List[Dict[str, float]] = []

    def get_fragmentation_ratio(self) -> float:
        """Calculate the current fragmentation ratio."""
        stats = torch.cuda.memory_stats(self.device)
        allocated = stats.get("allocated_bytes.all.current", 0)
        reserved = stats.get("reserved_bytes.all.current", 0)
        inactive_split = stats.get("inactive_split_bytes.all.current", 0)
        
        # Robust ratio as requested: non-releasable split vs total reserved
        if reserved == 0:
            return 0.0
        return inactive_split / reserved

    def log_allocator_stats(self):
        """Periodically log allocator metrics."""
        stats = torch.cuda.memory_stats(self.device)
        active_mb = stats.get("active_bytes.all.current", 0) / 1024**2
        reserved_mb = stats.get("reserved_bytes.all.current", 0) / 1024**2
        inactive_split_mb = stats.get("inactive_split_bytes.all.current", 0) / 1024**2
        retries = stats.get("num_alloc_retries", 0)
        ratio = self.get_fragmentation_ratio()
        
        logger.info(
            f"Allocator Stats: Active={active_mb:.2f}MB, Reserved={reserved_mb:.2f}MB, "
            f"Split={inactive_split_mb:.2f}MB, Ratio={ratio:.2%}, Retries={retries}"
        )
        
        # Update Prometheus metrics
        metrics.set_gauge("vram_usage_mb", reserved_mb, {"device_id": str(self.device)})
        metrics.set_gauge("vram_fragmentation_ratio", ratio, {"device_id": str(self.device)})

    def step(self, verbose: bool = False):
        """Perform a single monitoring and solving step."""
        if verbose:
            self.log_allocator_stats()

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        ratio = self.get_fragmentation_ratio()
        
        if ratio > self.config.trigger_ratio:
            self.high_frag_counter += 1
        else:
            self.high_frag_counter = 0

        # Trigger defragmentation if high fragmentation persists
        if self.high_frag_counter >= self.config.hysteresis_steps:
            self._defragment()
            self.high_frag_counter = 0

    def _defragment(self):
        """Perform safe defragmentation."""
        if self.empty_cache_count >= self.config.max_empty_cache_calls_per_epoch:
            logger.warning("Max empty_cache calls reached for this epoch.")
            return

        logger.info(f"Triggering defragmentation. Current frag ratio: {self.get_fragmentation_ratio():.2%}")
        
        # Sync and clear
        torch.cuda.synchronize(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        
        self.empty_cache_count += 1
        self.cooldown_counter = self.config.cooldown_steps
        
    @contextmanager
    def monitor(self):
        """Context manager for easy integration."""
        try:
            yield self
        finally:
            self.step()

    def reset_epoch_stats(self):
        """Reset counters at the start of a new epoch."""
        self.empty_cache_count = 0
        self.tensor_pool.reset()

def get_solver(device: str = "cuda:0") -> MemoryFragmentationSolver:
    """Factory function for the solver."""
    return MemoryFragmentationSolver(device=device)
