import torch
import torch.cuda
from typing import Optional, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorSwapper:
    """
    Manages CPU-GPU tensor swapping to optimize VRAM usage during training.
    
    Strategy:
    - Monitors GPU memory usage.
    - If usage exceeds threshold, offloads saved tensors (activations) to CPU.
    - Moves them back to GPU on demand during backward pass.
    """
    
    def __init__(self, threshold_percent: float = 0.8, device: str = "cuda"):
        """
        Args:
            threshold_percent (float): Memory usage threshold (0.0 to 1.0) to trigger swapping.
            device (str): Target device to monitor.
        """
        self.threshold = threshold_percent
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available():
            self.total_memory = torch.cuda.get_device_properties(self.device).total_memory
        else:
            self.total_memory = 1  # Avoid div by zero on CPU
            logger.warning("CUDA not available. TensorSwapper will be inactive.")

        self.swap_count = 0
        self.retrieve_count = 0

    def _get_memory_utilization(self) -> float:
        if self.device.type == 'cpu':
            return 0.0
        # memory_reserved is what matters for OOM prevention (allocated + cached)
        # However, for deciding if we *need* space, we check reserved.
        # If we want to offload when memory is *allocated* high, we check allocated.
        # Usually, fragmentation happens in reserved.
        # Let's check allocated / total.
        reserved = torch.cuda.memory_reserved(self.device)
        return reserved / self.total_memory

    def pack_hook(self, tensor: torch.Tensor) -> Any:
        """
        Hook called when a tensor is saved for backward.
        Decides whether to keep it on GPU or move to CPU.
        """
        # If tensor is not on our device, ignore (or handle gracefully)
        if tensor.device.type != self.device.type:
            return tensor

        utilization = self._get_memory_utilization()
        
        if utilization > self.threshold:
            # Swap to CPU
            self.swap_count += 1
            return tensor.cpu()
        
        # Keep on GPU
        return tensor

    def unpack_hook(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Hook called when a saved tensor is retrieved during backward.
        Moves it back to GPU if it was swapped.
        """
        if tensor.device.type == 'cpu' and self.device.type == 'cuda':
            self.retrieve_count += 1
            return tensor.to(self.device, non_blocking=True)
        
        return tensor

    def enable(self):
        """
        Returns a context manager that enables tensor swapping for the block.
        
        Usage:
            swapper = TensorSwapper()
            with swapper.enable():
                loss = model(input).sum()
                loss.backward()
        """
        return torch.autograd.graph.saved_tensors_hooks(self.pack_hook, self.unpack_hook)

    def reset_stats(self):
        self.swap_count = 0
        self.retrieve_count = 0
        
    def get_stats(self):
        return {
            "swaps": self.swap_count,
            "retrievals": self.retrieve_count
        }
