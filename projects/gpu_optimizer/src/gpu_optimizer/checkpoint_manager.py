import torch
import torch.nn as nn
from typing import List, Dict, Optional, Set
import logging
from torch.utils.checkpoint import checkpoint
from .cost_model import CostModel, LayerCost

logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    Automatically manages gradient checkpointing selection based on profiling data.
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda", sanitize_inplace: bool = True):
        self.model = model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cost_model = CostModel(device=self.device)
        self.checkpointed_layers: Set[str] = set()
        
        if sanitize_inplace:
            self.sanitize_inplace_ops()
            
    def sanitize_inplace_ops(self):
        """
        Sets inplace=False for all modules that have it (e.g. ReLU), 
        preventing 'variable modified by inplace operation' errors 
        during gradient checkpointing.
        """
        count = 0
        for module in self.model.modules():
            if hasattr(module, "inplace") and module.inplace:
                module.inplace = False
                count += 1
        logger.info(f"Sanitized {count} in-place operations for checkpointing safety.")
        
    def profile_and_optimize(self, 
                           sample_input: torch.Tensor, 
                           target_memory_reduction: float = 0.3,
                           max_compute_overhead: float = 0.2) -> List[str]:
        """
        Profiles the model and applies checkpointing to optimal layers.
        
        Args:
            sample_input: Input tensor for profiling
            target_memory_reduction: Target fraction of activation memory to save (0.0-1.0)
            max_compute_overhead: Max allowed increase in training time (0.0-1.0)
            
        Returns:
            List of layer names that were checkpointed.
        """
        logger.info("Profiling model for checkpointing candidates...")
        stats = self.cost_model.profile(self.model, sample_input)
        
        # Calculate totals
        total_time = sum(s.compute_time_ms for s in stats.values())
        total_act_mem = sum(s.activation_size_bytes for s in stats.values())
        
        if total_time == 0:
            logger.warning("Total compute time is 0. Cannot optimize.")
            return []

        # Rank layers by memory/compute ratio (Memory saved per ms of recompute)
        candidates = sorted(stats.values(), key=lambda x: x.memory_compute_ratio, reverse=True)
        
        current_saved_mem = 0
        current_overhead_ms = 0
        selected_layers = []
        
        overhead_budget_ms = total_time * max_compute_overhead
        memory_target_bytes = total_act_mem * target_memory_reduction
        
        logger.info(f"Optimization Targets:")
        logger.info(f"  Total Act Memory: {total_act_mem/1e6:.2f} MB")
        logger.info(f"  Target Savings:   {memory_target_bytes/1e6:.2f} MB ({target_memory_reduction:.0%})")
        logger.info(f"  Compute Budget:   {overhead_budget_ms:.2f} ms ({max_compute_overhead:.0%})")
        
        for layer in candidates:
            # Check if we met targets
            if current_saved_mem >= memory_target_bytes:
                break
                
            # Check budgets
            if current_overhead_ms + layer.compute_time_ms > overhead_budget_ms:
                continue
                
            # Select for checkpointing
            selected_layers.append(layer.name)
            current_saved_mem += layer.activation_size_bytes
            current_overhead_ms += layer.compute_time_ms
            
        logger.info(f"Selected {len(selected_layers)} layers for checkpointing.")
        logger.info(f"  Est. Savings:  {current_saved_mem/1e6:.2f} MB")
        logger.info(f"  Est. Overhead: {current_overhead_ms:.2f} ms")
        
        self.apply_checkpointing(selected_layers)
        return selected_layers

    def apply_checkpointing(self, layer_names: List[str]):
        """
        Wraps selected layers with torch.utils.checkpoint.
        """
        self.checkpointed_layers = set(layer_names)
        
        for name, module in self.model.named_modules():
            if name in self.checkpointed_layers:
                self._wrap_module(module)
                
    def _wrap_module(self, module: nn.Module):
        """
        Monkey-patches the module's forward method to use checkpointing.
        """
        original_forward = module.forward
        
        def checkpointed_forward(*args, **kwargs):
            # Checkpoint requires generated function to have inputs requiring grad
            # This is a simplification. For complex modules, specific handling is needed.
            return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)
        
        # Bind the new method to the instance
        # Using __get__ to bind method correctly as instance method
        # module.forward = checkpointed_forward.__get__(module, type(module))
        
        # Simpler approach: assign the function directly if it doesn't need 'self' 
        # (but it does, handled by closure over original_forward)
        module.forward = checkpointed_forward
