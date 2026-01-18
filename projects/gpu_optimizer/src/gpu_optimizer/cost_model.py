import torch
import torch.nn as nn
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class LayerCost:
    name: str
    compute_time_ms: float
    activation_size_bytes: int
    parameter_count: int
    
    @property
    def memory_compute_ratio(self) -> float:
        """Metric for checkpointing suitability: Higher is better."""
        if self.compute_time_ms == 0:
            return 0.0 # Avoid div by zero, but theoretically infinite
        return self.activation_size_bytes / self.compute_time_ms

class CostModel:
    """
    Profiles a PyTorch model to estimate compute time and memory usage per layer.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.layer_stats: Dict[str, LayerCost] = {}
        self.hooks = []
        
    def _get_shape_size(self, output) -> int:
        """Estimate size in bytes of a tensor or tuple of tensors."""
        if isinstance(output, torch.Tensor):
            return output.element_size() * output.nelement()
        elif isinstance(output, (tuple, list)):
            return sum(self._get_shape_size(x) for x in output)
        return 0

    def profile(self, model: nn.Module, sample_input: torch.Tensor, warmup: int = 3) -> Dict[str, LayerCost]:
        """
        Runs profiling on the model with the given input.
        """
        self.layer_stats = {}
        self.hooks = []
        
        # Helper to record timings
        start_times = {}
        
        def pre_hook(module, input, name):
            if self.device == "cuda":
                torch.cuda.synchronize()
            start_times[name] = time.time()

        def post_hook(module, input, output, name):
            if self.device == "cuda":
                torch.cuda.synchronize()
            duration = (time.time() - start_times.get(name, time.time())) * 1000
            
            act_size = self._get_shape_size(output)
            param_count = sum(p.numel() for p in module.parameters(recurse=False))
            
            self.layer_stats[name] = LayerCost(
                name=name,
                compute_time_ms=duration,
                activation_size_bytes=act_size,
                parameter_count=param_count
            )

        # Register hooks on leaf modules
        for name, module in model.named_modules():
            if len(list(module.children())) == 0: # Leaves only
                # Partial application to capture name
                self.hooks.append(module.register_forward_pre_hook(
                    lambda m, i, n=name: pre_hook(m, i, n)
                ))
                self.hooks.append(module.register_forward_hook(
                    lambda m, i, o, n=name: post_hook(m, i, o, n)
                ))

        model.to(self.device)
        model.eval()

        try:
            # Warmup
            with torch.no_grad():
                for _ in range(warmup):
                    model(sample_input)
            
            # Measurement
            with torch.no_grad():
                model(sample_input)
                
        finally:
            for h in self.hooks:
                h.remove()
            self.hooks = []

        return self.layer_stats
